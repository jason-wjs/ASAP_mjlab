import os
from typing import Dict

import torch
from loguru import logger


class MjlabBackend:
    """
    Backend engine wrapper for MJLab. Owns MuJoCo/Warp objects and exposes batched state I/O.
    """
    def __init__(self, device: str):
        self.device = device
        self.sim = None
        self.scene = None
        self.mj_model = None
        self.num_envs: int | None = None

    def build(self, *, xml_path: str, num_envs: int, env_spacing: float, sim_dt: float, sim_device: str,
              solver_iterations: int | None = None, njmax: int | None = None, nconmax: int | None = None,
              nan_guard_enable: bool | None = None):
        import mujoco
        from mjlab.scene.scene import Scene, SceneCfg
        from mjlab.entity.entity import EntityCfg
        from mjlab.sim.sim import Simulation, SimulationCfg, MujocoCfg
        try:
            from mjlab.utils.nan_guard import NanGuardCfg  # type: ignore
        except Exception:
            NanGuardCfg = None  # fallback if not available
        from mjlab.terrains.terrain_importer import TerrainImporterCfg

        def _robot_spec_fn() -> mujoco.MjSpec:
            return mujoco.MjSpec.from_file(xml_path)

        ent_cfg = EntityCfg(spec_fn=_robot_spec_fn)
        terr_cfg = TerrainImporterCfg(terrain_type="plane", env_spacing=env_spacing, num_envs=num_envs)
        scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=env_spacing, terrain=terr_cfg, entities={"robot": ent_cfg})

        scene = Scene(scene_cfg, device=str(sim_device))
        # Stable defaults: dt=0.002, solver iterations=50, sizes njmax=5000, nconmax=2000
        # Apply provided solver, size and nan-guard options (with safe defaults)
        iters = 50 if solver_iterations is None else int(solver_iterations)
        mj_cfg = MujocoCfg(timestep=sim_dt, iterations=iters)
        if NanGuardCfg is not None and nan_guard_enable is not None:
            nan_guard = NanGuardCfg(enabled=bool(nan_guard_enable))
            sim_cfg = SimulationCfg(mujoco=mj_cfg, njmax=njmax, nconmax=nconmax, nan_guard=nan_guard)
        else:
            sim_cfg = SimulationCfg(mujoco=mj_cfg, njmax=njmax, nconmax=nconmax)

        mj_model = scene.compile()
        # Ensure model size headroom as requested
        if njmax is not None:
            try:
                mj_model.njmax = int(njmax)
            except Exception:
                pass
        if nconmax is not None:
            try:
                mj_model.nconmax = int(nconmax)
            except Exception:
                pass
        sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=mj_model, device=str(sim_device))

        # Use mjlab's high-level bridges (Simulation.model/data) instead of raw
        # mjwarp structures to match the current mjlab API.
        scene.initialize(sim.mj_model, sim.model, sim.data)

        self.scene = scene
        self.sim = sim
        self.mj_model = mj_model
        self.num_envs = num_envs
        logger.info("MJLab backend build complete")

    def get_model(self):
        return self.mj_model

    def get_scene_env_origins(self) -> torch.Tensor:
        return self.scene.env_origins

    def forward(self):
        self.sim.forward()

    def step(self):
        self.sim.step()

    # -------- Raw state I/O --------
    def get_raw_state(self) -> Dict[str, torch.Tensor]:
        d = self.sim.data
        # Return raw MuJoCo batched views.
        # MuJoCo uses WXYZ for quaternions in xquat.
        return {
            "xpos": d.xpos,
            "xquat_wxyz": d.xquat,
            "cvel": d.cvel,
            "cfrc_ext": getattr(d, "cfrc_ext", None),
            "qpos": d.qpos,
            "qvel": d.qvel,
        }

    def set_root_state(self, env_ids: torch.Tensor, pos: torch.Tensor, quat_wxyz: torch.Tensor, lin_vel: torch.Tensor, ang_vel: torch.Tensor):
        d = self.sim.data
        d.qpos[env_ids, 0:3] = pos
        d.qpos[env_ids, 3:7] = quat_wxyz
        # 🔧 修复：MuJoCo qvel 格式是 [线速度, 角速度]
        d.qvel[env_ids, 0:3] = lin_vel  # 线速度在前
        d.qvel[env_ids, 3:6] = ang_vel  # 角速度在后

    def set_dof_state(self, env_ids: torch.Tensor, qpos_idx: torch.Tensor, qvel_idx: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor):
        d = self.sim.data
        d.qpos[env_ids[:, None], qpos_idx] = dof_pos
        d.qvel[env_ids[:, None], qvel_idx] = dof_vel

    def clear_applied_torques(self, v_adr: torch.Tensor):
        self.sim.data.qfrc_applied[:, v_adr] = 0.0

    def apply_torques(self, v_adr: torch.Tensor, tau: torch.Tensor):
        self.sim.data.qfrc_applied[:, v_adr] = tau
