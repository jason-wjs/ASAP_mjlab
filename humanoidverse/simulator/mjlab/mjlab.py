import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from loguru import logger

from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
from .mjlab_utils import normalize_name_for_mjlab
from .mjlab_randomization import init_dr_tensors
from humanoidverse.utils.torch_utils import quat_apply

class _SimpleTerrain:
    def __init__(self, env_origins: torch.Tensor):
        self.env_origins = env_origins

class MJLab(BaseSimulator):
    """
    MJLab simulator adapter implementing the BaseSimulator interface.

    Notes
    - Uses mjlab.scene.Scene + mjlab.sim.Simulation to build a batched MuJoCo simulation with a plane terrain.
    - Loads the robot from MJCF specified in config.robot.asset.xml_file under config.robot.asset.asset_root.
    - Exposes state tensors in the same shape/convention expected by envs.
    - Quaternions are exposed as XYZW to match env usage.
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        self.cfg = config
        self.sim_cfg = config.simulator.config
        self.robot_cfg = config.robot
        self.device = device

        # Will be set in setup() via backend
        self._backend = None
        self.mj_model = None

        # State tensors
        self.all_root_states: torch.Tensor | None = None
        self.robot_root_states: torch.Tensor | None = None
        self.base_quat: torch.Tensor | None = None  # XYZW
        self.dof_pos: torch.Tensor | None = None
        self.dof_vel: torch.Tensor | None = None
        self.contact_forces: torch.Tensor | None = None  # [num_envs, num_bodies, 3]

        self._rigid_body_pos: torch.Tensor | None = None
        self._rigid_body_rot: torch.Tensor | None = None  # XYZW
        self._rigid_body_vel: torch.Tensor | None = None
        self._rigid_body_ang_vel: torch.Tensor | None = None

        # Mappings
        self._body_list: List[str] | None = None  # ordered robot body names (excluding world)
        self._body_indices: List[int] | None = None  # indices in mj_model for bodies ordered per robot_cfg.body_names
        self._joint_ids: List[int] | None = None  # model joint ids in the same order as robot_cfg.dof_names
        self._joint_q_adr: torch.Tensor | None = None  # qpos addresses for joints
        self._joint_v_adr: torch.Tensor | None = None  # qvel addresses for joints

        # Domain-randomization related exposed buffers (expected by env obs getters)
        # Shapes:
        #  - _base_com_bias: [num_envs, 3]
        #  - _link_mass_scale: [num_envs, len(randomize_link_body_names)]
        #  - friction_coeffs: [num_envs, 1] (scalar per env)
        self._base_com_bias: torch.Tensor | None = None
        self._link_mass_scale: torch.Tensor | None = None
        self.friction_coeffs: torch.Tensor | None = None

        self.num_envs = None
        self.num_bodies = None
        self.num_dof = None
        self.dof_names: List[str] | None = None
        self.body_names: List[str] | None = None

    def set_headless(self, headless):
        super().set_headless(headless)

    def setup(self):
        # Timing and solver params configurable from YAML
        # Fallbacks: dt from fps, substeps=1, iterations/limits None
        sim_cfg = self.sim_cfg.sim
        fps = getattr(sim_cfg, "fps", 500)
        default_dt = 1.0 / float(fps)
        self.sim_dt = float(getattr(sim_cfg, "dt", default_dt))
        self._substeps = int(getattr(sim_cfg, "substeps", 1))
        self._solver_iterations = int(getattr(sim_cfg, "solver_iterations", 50))
        self._njmax = getattr(sim_cfg, "njmax", None)
        self._nconmax = getattr(sim_cfg, "nconmax", None)
        print("njmax =", self._njmax, "nconmax =", self._nconmax)

        # nan_guard from YAML: simulator.config.sim.nan_guard.enable
        try:
            self._nan_guard_enable = bool(getattr(sim_cfg.nan_guard, "enable"))  # type: ignore[attr-defined]
        except Exception:
            self._nan_guard_enable = None

        # Build scene with plane terrain and a single robot entity from MJCF
        # Resolve MJCF path robustly with multiple candidates
        asset_root = str(self.robot_cfg.asset.asset_root)
        xml_file = str(self.robot_cfg.asset.xml_file) 

        candidates = []
        # 1) Absolute xml_file wins
        if os.path.isabs(xml_file):
            candidates.append(xml_file)
        # 2) Provided asset_root + xml_file (may itself be absolute or relative)
        candidates.append(os.path.join(asset_root, xml_file))
        # 3) Resolve relative to CWD
        candidates.append(os.path.join(os.getcwd(), asset_root, xml_file))
        # 4) If asset_root accidentally embeds project name (e.g., "humanoidverse/data/robots"),
        #    also try stripping it to "data/robots"
        try_root = asset_root
        if try_root.startswith("humanoidverse/"):
            try_root = try_root[len("humanoidverse/"):]
        candidates.append(os.path.join(os.getcwd(), try_root, xml_file))
        # 5) Directly try project-local data/robots
        candidates.append(os.path.join(os.getcwd(), "data/robots", xml_file))
        # 6) User-provided absolute path for testing in request
        candidates.append("/home/wujs/Projects/PBHC_Adam/humanoidverse/data/robots/adam_sp/adam_sp.xml")

        xml_path = None
        for p in candidates:
            if p and os.path.exists(p):
                xml_path = p
                break
        if xml_path is None:
            raise FileNotFoundError(
                f"Robot MJCF not found. Tried: {candidates}"
            )

        # Backend build
        from .mjlab_backend import MjlabBackend
        self._backend = MjlabBackend(device=str(self.sim_device))
        self._backend.build(xml_path=xml_path,
                            num_envs=self.cfg.num_envs,
                            env_spacing=self.sim_cfg.scene.env_spacing,
                            sim_dt=self.sim_dt,
                            sim_device=str(self.sim_device),
                            solver_iterations=self._solver_iterations,
                            njmax=self._njmax,
                            nconmax=self._nconmax,
                            nan_guard_enable=self._nan_guard_enable
                            )
        self.mj_model = self._backend.get_model()

        # Terrain origins as torch tensor on device
        env_origins = self._backend.get_scene_env_origins().to(self.sim_device).to(torch.float32)
        self.terrain = _SimpleTerrain(env_origins)

        # Device used for tensors
        self.device = self.sim_device

        logger.info("MJLab simulator setup complete")

    def setup_terrain(self, mesh_type):
        # Scene already includes plane terrain via TerrainImporterCfg.
        if mesh_type not in [None, "plane"]:
            logger.warning(f"setup_terrain: only 'plane' is supported for mjlab now (got '{mesh_type}').")

    def load_assets(self):
        # Build mappings from model using configured dof/body names
        model = self.mj_model
        import mujoco
        all_body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        all_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]

        # Create a list of all body names from the model, excluding the world body
        model_body_names = [name for name in all_body_names[1:] if name]

        # Create a list of all joint names from the model
        model_joint_names = [name for name in all_joint_names if name]

        # --- Body Mapping ---
        self.body_names = list(self.robot_cfg.body_names)
        self.num_bodies = len(self.body_names)
        self._body_list = list(self.robot_cfg.body_names) # For compatibility with find_rigid_body_indice

        body_indices = []
        body_map_info = []
        for name in self.body_names:
            # Try to find the body with potential prefixes
            found = False
            for prefix in ["", "robot/"]: # Add other potential prefixes if needed
                candidate_name = f"{prefix}{name}"
                if candidate_name in all_body_names:
                    idx = all_body_names.index(candidate_name)
                    body_indices.append(idx)
                    body_map_info.append(f"  '{name}' -> '{candidate_name}' (idx: {idx})")
                    found = True
                    break
            if not found:
                raise ValueError(f"Body name '{name}' not found in MuJoCo model. Searched variants. Model bodies: {model_body_names}")

        self._body_indices = body_indices
        logger.info("MJLab body mapping (config -> model):")
        for line in body_map_info:
            logger.info(line)

        # --- DOF/Joint Mapping ---
        self.dof_names = list(self.robot_cfg.dof_names)
        self.num_dof = len(self.dof_names)

        joint_ids = []
        joint_map_info = []
        logger.info("--- MJLab Joint Mapping ---")
        for name in self.dof_names:
            # Try to find the joint with potential prefixes
            found = False
            for prefix in ["", "robot/"]: # Add other potential prefixes if needed
                candidate_name = f"{prefix}{name}"
                try:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate_name)
                    if jid >= 0:
                        dof_adr = model.jnt_dofadr[jid]
                        logger.info(f"  - Config name: '{name}' -> Model name: '{candidate_name}' | ID: {jid}, Qvel Addr: {dof_adr}")
                        joint_ids.append(jid)
                        found = True
                        break
                except ValueError:
                    continue # Try next prefix
            if not found:
                raise ValueError(f"Joint (dof) name '{name}' not found in MuJoCo model. Searched variants. Model joints: {model_joint_names}")

        self._joint_ids = joint_ids
        # logger.info("MJLab joint mapping (config -> model):")
        # for line in joint_map_info:
        #     logger.info(line)


        # Compute qpos and qvel addresses for the joints
        jnt_qposadr = torch.tensor([int(model.jnt_qposadr[i]) for i in joint_ids], device=self.device, dtype=torch.long)
        jnt_dofadr = torch.tensor([int(model.jnt_dofadr[i]) for i in joint_ids], device=self.device, dtype=torch.long)
        self._joint_q_adr = jnt_qposadr
        self._joint_v_adr = jnt_dofadr

        # basic assertions align with config sizes
        assert self.num_dof == len(self.robot_cfg.dof_names), "Mismatch in DOF count"
        assert self.num_bodies == len(self.robot_cfg.body_names), "Mismatch in body count"
        assert len(self._body_indices) == self.num_bodies, "Not all configured bodies were found in the model"
        assert len(self._joint_ids) == self.num_dof, "Not all configured joints were found in the model"

    def create_envs(self, num_envs, env_origins, base_init_state):
        # Cache basics
        self.num_envs = num_envs
        self.env_origins = env_origins  # torch tensor [num_envs, 3]

        # Initialize root state and joints from config.init_state
        # Root pose
        base_pos = torch.tensor(self.robot_cfg.init_state.pos, device=self.device, dtype=torch.float32).view(1, 3)
        base_quat_xyzw = torch.tensor(self.robot_cfg.init_state.rot, device=self.device, dtype=torch.float32).view(1, 4)
        base_quat_wxyz = base_quat_xyzw[..., [3, 0, 1, 2]]

        # Broadcast to all envs and add env origins
        base_pos = base_pos.repeat(self.num_envs, 1) + env_origins[:, :3]
        base_quat_wxyz = base_quat_wxyz.repeat(self.num_envs, 1)

        # Joint default angles
        default_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        for i, name in enumerate(self.dof_names):
            default_pos[:, i] = float(self.robot_cfg.init_state.default_joint_angles[name])
        default_vel = torch.zeros_like(default_pos)

        # Write initial state via backend and forward
        env_ids = torch.arange(self.num_envs, device=self.device)
        zeros = torch.zeros_like(base_pos)
        self._backend.set_root_state(env_ids, base_pos, base_quat_wxyz, zeros, zeros)
        self._backend.set_dof_state(env_ids, self._joint_q_adr, self._joint_v_adr, default_pos, default_vel)
        self._backend.forward()

        # Initialize exposure tensors
        self._init_tensors()

        # Initialize DR exposure buffers (no-op defaults for now)
        try:
            link_names_len = 0
            if hasattr(self.cfg, "domain_rand") and hasattr(self.cfg.domain_rand, "randomize_link_body_names"):
                link_names_len = len(self.cfg.domain_rand.randomize_link_body_names)
            elif hasattr(self.robot_cfg, "randomize_link_body_names") and self.robot_cfg.randomize_link_body_names is not None:
                link_names_len = len(self.robot_cfg.randomize_link_body_names)
        except Exception:
            link_names_len = 0
        (_base_com_bias,
         _link_mass_scale,
         friction_coeffs,
         restitution_coeffs,
         _ground_friction_values) = init_dr_tensors(self.num_envs, self.num_bodies, link_names_len, device=self.device)
        self._base_com_bias = _base_com_bias
        self._link_mass_scale = _link_mass_scale
        self.friction_coeffs = friction_coeffs.view(self.num_envs, 1)
        self.restitution_coeffs = restitution_coeffs.view(self.num_envs, 1, 1)
        self._ground_friction_values = _ground_friction_values
        return [], []

    def _init_tensors(self):
        # Allocate fixed-size tensors on device
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.all_root_states = torch.zeros(self.num_envs, 13, device=self.device)
        self.robot_root_states = torch.zeros(self.num_envs, 13, device=self.device)
        self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self._rigid_body_pos = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self._rigid_body_rot = torch.zeros(self.num_envs, self.num_bodies, 4, device=self.device)
        self._rigid_body_vel = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self._rigid_body_ang_vel = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)

    def get_dof_limits_properties(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Follow Genesis: use config to populate hard/soft limits and torque limits
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        self.dof_pos_limits_termination = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device)

        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = float(self.robot_cfg.dof_pos_lower_limit_list[i])
            self.hard_dof_pos_limits[i, 1] = float(self.robot_cfg.dof_pos_upper_limit_list[i])
            self.dof_pos_limits[i, 0] = float(self.robot_cfg.dof_pos_lower_limit_list[i])
            self.dof_pos_limits[i, 1] = float(self.robot_cfg.dof_pos_upper_limit_list[i])
            self.dof_vel_limits[i] = float(self.robot_cfg.dof_vel_limit_list[i])
            self.torque_limits[i] = float(self.robot_cfg.dof_effort_limit_list[i])

            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.reward_limit.soft_dof_pos_limit
            
            # termination limits (similar to IsaacGym implementation)
            self.dof_pos_limits_termination[i, 0] = m - 0.5 * r * self.cfg.termination_scales.termination_close_to_dof_pos_limit
            self.dof_pos_limits_termination[i, 1] = m + 0.5 * r * self.cfg.termination_scales.termination_close_to_dof_pos_limit

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name: str) -> int:
        """
        Find body index in the exposed _rigid_body_* tensors.
        
        Returns the index in self.body_names (the configured body order),
        NOT the MuJoCo model body index.
        """
        try:
            # Return the index in body_names, which is the index in _rigid_body_* tensors
            return self.body_names.index(body_name)
        except ValueError:
            # If not found, raise an error with a helpful message.
            raise ValueError(
                f"Body '{body_name}' not found in configured `robot.body_names` list. "
                f"Available bodies: {self.body_names}"
            )

    def prepare_sim(self):
        # Advance once and populate tensors
        self._backend.step()
        self.refresh_sim_tensors()

    def refresh_sim_tensors(self):
        # Read latest state from backend into exposed tensors
        raw = self._backend.get_raw_state()

        # Root state: use the explicitly configured root_body as the reference
        root_body_name = getattr(self.robot_cfg, "root_body", self.body_names[0])
        if root_body_name not in self.body_names:
            # Fallback to the first body if root_body is not in the configured list
            logger.warning(f"Configured root_body '{root_body_name}' not in body_names. Falling back to '{self.body_names[0]}'.")
            root_body_name = self.body_names[0]
            
        root_body_config_idx = self.body_names.index(root_body_name)
        root_body_idx = self._body_indices[root_body_config_idx]

        base_pos = raw["xpos"][:, root_body_idx]
        
        # Backend provides quaternions in WXYZ format. Convert to XYZW for humanoidverse.
        base_quat_wxyz = raw["xquat_wxyz"][:, root_body_idx]
        base_quat_xyzw = base_quat_wxyz[..., [1, 2, 3, 0]]

        # debug: 从 qvel 直接读取 (世界坐标系)
        qvel_lin_vel = raw["qvel"][:, 0:3]
        qvel_ang_vel = raw["qvel"][:, 3:6]
        
        # debug: 从 cvel 读取并转换 (体坐标系→世界坐标系)
        base_vel_local = raw["cvel"][:, root_body_idx]  # [wx, wy, wz, vx, vy, vz] in body frame
        local_ang = base_vel_local[..., 0:3]
        local_lin = base_vel_local[..., 3:6]
        # Rotate to world frame using body orientation
        cvel_ang_vel = quat_apply(base_quat_xyzw, local_ang)
        cvel_lin_vel = quat_apply(base_quat_xyzw, local_lin)
        
        # debug: 单环境时打印,避免日志爆炸
        if self.num_envs == 1 and not hasattr(self, '_velocity_diff_logged'):
            lin_diff = torch.abs(qvel_lin_vel - cvel_lin_vel).max()
            ang_diff = torch.abs(qvel_ang_vel - cvel_ang_vel).max()
            if lin_diff > 1e-4 or ang_diff > 1e-4:
                logger.warning(f"🔴 Root velocity mismatch! qvel vs cvel:")
                logger.warning(f"  qvel lin: {qvel_lin_vel[0]}")
                logger.warning(f"  cvel lin: {cvel_lin_vel[0]}")
                logger.warning(f"  lin diff: {lin_diff:.6f}")
                logger.warning(f"  qvel ang: {qvel_ang_vel[0]}")
                logger.warning(f"  cvel ang: {cvel_ang_vel[0]}")
                logger.warning(f"  ang diff: {ang_diff:.6f}")
                logger.warning(f"  Consider using qvel directly for root velocity!")
                self._velocity_diff_logged = True
            elif not hasattr(self, '_velocity_match_logged'):
                logger.info(f"✅ Root velocity match: qvel vs cvel (diff < 1e-4)")
                self._velocity_match_logged = True
        
        # 🔧 TODO: 如果发现差异,尝试改用 qvel
        # base_lin_vel = qvel_lin_vel
        # base_ang_vel = qvel_ang_vel
        
        # 当前仍使用 cvel 方法
        base_ang_vel = cvel_ang_vel
        base_lin_vel = cvel_lin_vel

        self.base_quat[:] = base_quat_xyzw
        self.all_root_states[:, 0:3] = base_pos
        self.all_root_states[:, 3:7] = base_quat_xyzw
        self.all_root_states[:, 7:10] = base_lin_vel
        self.all_root_states[:, 10:13] = base_ang_vel
        self.robot_root_states[:] = self.all_root_states

        # Joints
        self.dof_pos[:] = raw["qpos"][:, self._joint_q_adr]
        self.dof_vel[:] = raw["qvel"][:, self._joint_v_adr]

        # Bodies (positions, rotations, velocities, angular velocities)
        # Reorder to match humanoidverse config body order
        idx = torch.tensor(self._body_indices, device=self.device, dtype=torch.long)
        self._rigid_body_pos[:] = raw["xpos"].index_select(1, idx)
        
        # Backend provides WXYZ, convert to XYZW
        _xquat_wxyz = raw["xquat_wxyz"].index_select(1, idx)
        self._rigid_body_rot[:] = _xquat_wxyz[..., [1, 2, 3, 0]]

        # Convert cvel (ang first then lin) from body-frame to world-frame
        cvel = raw["cvel"].index_select(1, idx)
        local_ang_all = cvel[..., 0:3]
        local_lin_all = cvel[..., 3:6]
        q_all = self._rigid_body_rot  # XYZW
        self._rigid_body_ang_vel[:] = quat_apply(q_all.reshape(-1, 4), local_ang_all.reshape(-1, 3)).view(self.num_envs, self.num_bodies, 3)
        self._rigid_body_vel[:] = quat_apply(q_all.reshape(-1, 4), local_lin_all.reshape(-1, 3)).view(self.num_envs, self.num_bodies, 3)

        # Contact forces: mimic IsaacGym net contact force (per-body force in world frame)
        # Attempt to use MuJoCo's cfrc_ext if available; else zero.
        if raw.get("cfrc_ext", None) is not None:
            # cfrc_ext: [B, nbody, 6]
            cfrc = raw["cfrc_ext"].index_select(1, idx)
            self.contact_forces[:] = cfrc[..., 0:3]
        else:
            # keep zeros; env can still run without contacts
            if (self.contact_forces is None) or (self.contact_forces.numel() == 0):
                self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)

    def apply_torques_at_dof(self, torques: torch.Tensor):
        # Apply generalized torques at joint velocity addresses via backend
        self._backend.clear_applied_torques(self._joint_v_adr)
        self._backend.apply_torques(self._joint_v_adr, torques)

    def set_actor_root_state_tensor(self, set_env_ids: torch.Tensor, root_states: torch.Tensor):
        # root_states is [num_envs, 13] with XYZW quat
        env_ids = set_env_ids
        base_pos = root_states[env_ids, 0:3]
        base_quat_xyzw = root_states[env_ids, 3:7]
        base_quat_wxyz = base_quat_xyzw[..., [3, 0, 1, 2]]
        base_lin_vel = root_states[env_ids, 7:10]
        base_ang_vel = root_states[env_ids, 10:13]
        self._backend.set_root_state(env_ids, base_pos, base_quat_wxyz, base_lin_vel, base_ang_vel)

    def set_dof_state_tensor(self, set_env_ids: torch.Tensor, dof_states: torch.Tensor):
        env_ids = set_env_ids
        dof_pos = dof_states.view(self.num_envs, -1, 2)[env_ids, :, 0]
        dof_vel = dof_states.view(self.num_envs, -1, 2)[env_ids, :, 1]
        self._backend.set_dof_state(env_ids, self._joint_q_adr, self._joint_v_adr, dof_pos, dof_vel)

    def simulate_at_each_physics_step(self):
        # Advance simulation with internal substeps for stability
        sub = int(getattr(self, "_substeps", 1))
        for _ in range(max(1, sub)):
            self._backend.step()

    def setup_viewer(self):
        # Viewer selection based on mjlab modes: auto | viser | native | none
        # Default to auto if not configured
        mode = None
        render_interval = 1
        try:
            mode = getattr(self.sim_cfg.sim.viewer, "mode")  # type: ignore[attr-defined]
            render_interval = int(getattr(self.sim_cfg.sim.viewer, "render_interval", 1))  # type: ignore[attr-defined]
        except Exception:
            mode = None
        if mode is None:
            mode = "auto"

        env_idx = 0

        if mode in ("auto", "viser"):
            try:
                import viser  # type: ignore
                from mjlab.viewer.viser_scene import ViserMujocoScene  # type: ignore

                self._viser_server = viser.ViserServer(label="humanoidverse-mjlab")
                self._viser_scene = ViserMujocoScene.create(
                    server=self._viser_server,
                    mj_model=self.mj_model,
                    num_envs=int(self.num_envs if self.num_envs is not None else 1),
                )

                class _Viewer:
                    pass

                self.viewer = _Viewer()
                self.viewer.env_idx = env_idx
                self.viewer.server = self._viser_server
                self.viewer.scene = self._viser_scene
                self.viewer.mode = "viser"
                self.viewer.render_interval = render_interval
                self._render_counter = 0

                # Draw an initial frame
                try:
                    raw = self._backend.get_raw_state()
                    qpos0 = raw["qpos"][env_idx].detach().cpu().numpy()
                    self._viser_scene.add_ghost_mesh(qpos0, self.mj_model, alpha=1.0)
                except Exception:
                    pass
                return
            except Exception as e:
                if mode == "viser":
                    logger.warning(f"Viser viewer requested but unavailable: {e}")
                    self.viewer = None
                    return
                # else fall through to native

        if mode in ("auto", "native"):
            try:
                import mujoco
                import mujoco.viewer
                # Use MJLab sim's mj_model and mj_data
                mj_model = self._backend.sim.mj_model  # type: ignore[attr-defined]
                mj_data = self._backend.sim.mj_data  # type: ignore[attr-defined]
                viewer_handle = mujoco.viewer.launch_passive(
                    mj_model,
                    mj_data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
                if viewer_handle is None:
                    raise RuntimeError("Failed to launch MuJoCo native viewer")

                class _Viewer:
                    pass

                self.viewer = _Viewer()
                self.viewer.env_idx = env_idx
                self.viewer.native = viewer_handle
                self.viewer.mj_model = mj_model
                self.viewer.mj_data = mj_data
                self.viewer.mode = "native"
                self.viewer.render_interval = render_interval
                self._render_counter = 0
                return
            except Exception as e:
                if mode == "native":
                    logger.warning(f"Native viewer requested but unavailable: {e}")
                    self.viewer = None
                    return

        # none or failed
        if mode not in ("auto", "viser", "native"):
            logger.info(f"Viewer mode set to '{mode}', viewer disabled.")
        else:
            logger.warning("Viewer setup failed; running headless.")
        self.viewer = None

    def render(self, sync_frame_time=True):
        if not hasattr(self, "viewer") or self.viewer is None:
            return
        # Throttle render frequency
        self._render_counter = (self._render_counter + 1) % max(1, getattr(self.viewer, "render_interval", 1))
        if self._render_counter != 0:
            return
        mode = getattr(self.viewer, "mode", None)
        env_idx = getattr(self.viewer, "env_idx", 0)
        if mode == "viser" and hasattr(self, "_viser_scene") and self._viser_scene is not None:
            # Viser expects CPU/NumPy data
            qpos = self._backend.get_raw_state()["qpos"][env_idx].detach().cpu().numpy()
            try:
                self._viser_scene.add_ghost_mesh(qpos, self.mj_model, alpha=1.0)
            except Exception:
                pass
        elif mode == "native" and hasattr(self.viewer, "native"):
            try:
                import mujoco
                raw = self._backend.get_raw_state()
                mj_model = self.viewer.mj_model
                mj_data = self.viewer.mj_data
                mj_data.qpos[:] = raw["qpos"][env_idx].detach().cpu().numpy()
                mj_data.qvel[:] = raw["qvel"][env_idx].detach().cpu().numpy()
                mujoco.mj_forward(mj_model, mj_data)
                self.viewer.native.sync(state_only=True)
            except Exception:
                pass
        return

    # ----- Optional debug drawing API used by motion_tracking -----
    def clear_lines(self):
        # No-op for now; Viser visualizer manages its own scene.
        pass

    def draw_sphere(self, pos: torch.Tensor, radius: float, color: tuple, env_id: int, pos_id: int):
        # Optional visualization of marker spheres in Viser.
        # mjlab DebugVisualizer does not provide add_sphere; skip gracefully.
        try:
            _ = pos  # keep signature; no-op for now
            _ = radius
            _ = color
            _ = env_id
            _ = pos_id
        except Exception:
            pass

    @property
    def dof_state(self):
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)
