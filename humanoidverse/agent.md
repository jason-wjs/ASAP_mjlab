# agent.md

## 1. Project Overview

This repository implements a modular humanoid-robotics and embodied-AI stack focused on learning locomotion and motion-tracking policies in high-performance physics simulators such as Isaac Gym, Isaac Sim, Genesis, and MJLab. At its core, the system integrates humanoid simulation, terrain generation, motion-library–based reference tracking, and reinforcement/imitation learning via PPO-style agents and specialized variants (e.g., Delta-A methods, world-model augmentation, decoupled controllers, force-space control). Training and evaluation pipelines are orchestrated through Hydra/OmegaConf configurations, which bind together simulator backends, robot assets, task environments, reward functions, observation definitions, domain-randomization schemes, and agent algorithms into fully reproducible experiments.

The project is currently in a transition phase: HumanoidVerse already provides complete and mature support for IsaacGym, Genesis, and IsaacSim, and development has now moved to integrating a fourth backend—MJLab, a new high-performance MuJoCo-Warp–based simulator. The MJLab implementation resides in the external repository located at /home/wujs/Projects/mjlab, and the current engineering effort focuses on designing a clean adapter layer, unifying data structures (DOF layout, quaternion conventions, root pose updates), and ensuring full compatibility between HumanoidVerse’s training stack and the MJLab simulation runtime.

---

## 2. Repository Structure Overview

High-level structure (only major development/experiment-related paths):

- `train_agent.py` — Main training entry point (Hydra-based).
- `eval_agent.py` — Main evaluation and policy export entry point.
- `config/`
  - `base/`
    - `base.yaml` — Global defaults for training (seed, dirs, wandb flags, simulator type, etc.).
    - `base_eval.yaml` — Global defaults for evaluation (eval dirs, overrides).
    - `structure.yaml` — Top-level config structure placeholders (`algo`, `env`, `robot`, `terrain`).
    - `hydra.yaml` — Hydra runtime/sweep directory settings.
  - `algo/` — Algorithm configs (e.g., `ppo.yaml`, `ppo_train_delta_a.yaml`).
  - `env/` — Environment/task configs (e.g., `base_task.yaml`, `locomotion*.yaml`, `motion_tracking*.yaml`, `delta_a_*loop.yaml`).
  - `simulator/` — Simulator configs (`isaacgym.yaml`, `isaacsim.yaml`, `genesis.yaml`, `mjlab.yaml`, `mujoco.yaml`).
  - `robot/` — Robot configuration templates (`robot_base.yaml`) and concrete robot definitions (`g1/g1_29dof_anneal_23dof.yaml`).
  - `terrain/` — Terrain-related configs for simulation.
  - `obs/` — Observation configs for different tasks (`legged_obs.yaml`, `loco`, `motion_tracking`, `delta_a`).
  - `rewards/` — Reward configuration modules for locomotion and motion tracking.
  - `domain_rand/` — Domain randomization configs (`domain_rand_base.yaml`, `NO_domain_rand.yaml`).
  - `opt/` — Optimization/logging configs (e.g., wandb, eval analysis).
  - `exp/` — Experiment-level presets (naming, evaluation plotting, wandb, record).
- `agents/`
  - `base_algo/` — `base_algo.py` (BaseAlgo interface).
  - `ppo/` — `ppo.py` (PPO implementation).
  - `delta_a/` — `train_delta_a.py` (PPODeltaA, policy-with-deltaA training).
  - `dagger/`, `decouple/`, `delta_dynamics/`, `force_control/`, `mppi/` — Specialized RL/IL/world-model/mppi/force estimation agents.
  - `modules/` — Network and RL building blocks:
    - `ppo_modules.py` — PPOActor/PPOCritic modules.
    - `world_models.py`, `encoder_modules.py`, `modules.py`, `data_utils.py` — World models, encoders, rollout storage, etc.
  - `callbacks/` — RL callbacks for logging and analysis plots (locomotion, motion tracking, motion tracking open-loop, force, etc.).
  - `ppo_locomanip.py` — PPO variant for loco-manipulation.
- `envs/`
  - `base_task/` — `base_task.py` (BaseTask: generic RL task built on a `BaseSimulator`).
  - `legged_base_task/` — `legged_robot_base.py` (legged humanoid task base).
  - `locomotion/` — `locomotion.py` (LeggedRobotLocomotion).
  - `motion_tracking/` — `motion_tracking.py` (LeggedRobotMotionTracking).
  - `delta_a/` — `delta_a_*loop.py` (Delta-A closed/open-loop motion tracking variants).
  - `env_utils/` — Terrain utilities, history handling, command generator, generic helpers and visualization.
- `simulator/`
  - `base_simulator/` — `base_simulator.py` (BaseSimulator interface).
  - `isaacgym/` — `isaacgym.py`, `isaacgym_hoi.py` (Isaac Gym backend for humanoid and HOI).
  - `isaacsim/` — Isaac Sim backend and config helpers.
  - `genesis/` — Genesis backend and utilities.
  - `mjlab/` — MJLab backend (`mjlab.py`, `mjlab_backend.py`, `mjlab_randomization.py`, `mjlab_utils.py`).
- `utils/`
  - `config_utils.py` — OmegaConf resolvers registration.
  - `common.py` — Seeding, CLI utilities, helper print functions.
  - `helpers.py` — Config preprocessing, observation parsing, misc utilities.
  - `inference_helpers.py` — Exporting policies to TorchScript / ONNX, inference helper utilities.
  - `logging.py` — Hydra–Loguru bridge and logging helpers.
  - `torch_utils.py`, `math.py`, `terrain.py` — Torch math utilities, terrain helpers.
  - `motion_lib/` — Motion library (skeleton, humanoid motion loading).
  - `average_meters.py` — Metric accumulators for logging.
- `data/`
  - `motions/` — Humanoid motion data (e.g., `g1_29dof_anneal_23dof`, raw SMPL).
  - `robots/` — Robot assets (e.g., `g1` model files).
- `envs/` (package `humanoidverse.envs` via `__init__.py`) — Makes envs importable as `humanoidverse.envs.*`.
- `agents/`, `simulator/`, `utils/` similarly form the `humanoidverse` package when installed.
- `envs/`, `agents/`, `simulator/`, `utils/`, `config/`, `data/` — main vertical slices representing env definitions, algorithms, simulators, utilities, configuration, and assets.

---

## 3. Modules & Responsibilities

### 3.1 Top-Level Execution Flow

- **`train_agent.py`**
  - Responsibilities:
    - Entry point for training an RL agent.
    - Uses Hydra (`@hydra.main(config_path="config", config_name="base")`) to build a unified config.
    - Resolves simulator type and, for IsaacSim, launches `omni.isaac.lab.app.AppLauncher` with CLI arguments derived from config.
    - Configures logging (Loguru + HydraLoggerBridge), W&B, device selection, and experiment directories.
    - Calls `pre_process_config(config)` to finalize config structure (e.g., fill derived fields, attach paths).
    - Instantiates `env: BaseTask` from `config.env`, then `algo: BaseAlgo` from `config.algo`.
    - Saves the resolved (unresolved Hydra) config to disk (`experiment_dir/config.yaml`).
    - Optionally loads a checkpoint via `algo.load`, then runs `algo.learn()` to train.
    - For IsaacSim, closes the app after training.
  - Interactions:
    - Depends on `config/*` modules (for algo/env/robot/simulator).
    - Interacts with `BaseTask` and `BaseAlgo` implementations, which in turn talk to simulators and policies.
    - Uses `utils.common.seeding` (commented in current snippet but conceptually part of flow) and W&B for logging.

- **`eval_agent.py`**
  - Responsibilities:
    - Entry point for evaluation and optional policy export.
    - Uses Hydra (`config_name="base_eval"`) and merges evaluation overrides with training config, if a checkpoint is provided.
    - Reconstructs config from `checkpoint` directory’s `config.yaml` where possible, merges `eval_overrides` and CLI overrides.
    - Sets up logging (Loguru + Hydra logger bridge).
    - Handles simulator-specific initialization (IsaacSim vs IsaacGym).
    - Calls `pre_process_config(config)` to ensure env/algo/robot configs are in the right shape for instantiation.
    - Creates env and algo, loads weights using `algo.load(config.checkpoint)`.
    - Optionally exports policy to JIT/ONNX using `inference_helpers` and example obs.
    - Runs `algo.evaluate_policy()` for evaluation or rollout recording.
  - Interactions:
    - Uses the same env and algo modules as training but with `eval_overrides` (e.g., `num_envs=1`, `headless=False`).
    - Writes evaluation config to `eval_log_dir/config.yaml`.
    - Optional ONNX export for integration with other stacks (e.g., external controllers).

### 3.2 Configuration Modules

- **`config/base/*.yaml`**
  - Responsibilities:
    - `base.yaml`: Top-level training defaults (seed, `num_envs`, `project_name`, dirs, `use_wandb`, `sim_type`, `output_dir`, `eval_overrides`).
    - `base_eval.yaml`: Top-level eval defaults (`eval_log_dir`, Hydra run dir).
    - `structure.yaml`: Declares required top-level keys (`algo`, `env`, `robot`, `terrain`) as `???` to be filled by experiment configs.
    - `hydra.yaml`: Hydra output dir layout (training logs saved under `save_dir`).
  - Interactions:
    - Combined with `config/exp`, `config/algo`, `config/env`, `config/robot`, `config/simulator`, etc., at runtime by Hydra.
    - `eval_overrides` is merged into config for evaluation or at inference.

- **`config/env/*.yaml`**
  - Responsibilities:
    - For each task type (base_task, locomotion, motion_tracking, delta_a), defines:
      - `_target_` of the environment class (e.g., `humanoidverse.envs.locomotion.locomotion.LeggedRobotLocomotion`).
      - Environment-specific fields: `num_envs`, `max_episode_length_s`, domain randomization flags, etc.
      - Nesting for `simulator`, `robot`, `terrain`, `obs`, `rewards`, `domain_rand` references.
    - Special variants for specific backends (e.g., `locomotion_genesis.yaml`, `locomotion_mjlab.yaml`).
  - Interactions:
    - Instantiated by Hydra in `train_agent.py`/`eval_agent.py`.
    - Passed into `BaseTask` / `LeggedRobotBase` constructors.

- **`config/algo/*.yaml`**
  - Responsibilities:
    - Define algorithm hyperparameters for PPO, PPODeltaA, etc.:
      - `num_steps_per_env`, `num_learning_iterations`, `gamma`, `lam`, `learning_rates`, `entropy_coef`, etc.
      - Model architecture references (`module_dict.actor`, `module_dict.critic`) for `PPOActor`/`PPOCritic`.
      - Logging and checkpoint intervals.
  - Interactions:
    - Instantiated into `algo: BaseAlgo` (`agents/ppo/ppo.PPO`, `agents/delta_a/train_delta_a.PPODeltaA`, etc.).
    - Accessed inside `PPO` for storage sizing, network construction, and training loops.

- **`config/robot/*.yaml`**
  - Responsibilities:
    - `robot_base.yaml` declares generic structure (dof/obs sizes, asset parameters) with some `???` placeholders.
    - `g1/g1_29dof_anneal_23dof.yaml` concretizes G1 humanoid specifics:
      - DOF counts, DOF names, body names, effort/velocity limits, PD gains, armature/friction, etc.
      - Body groups (lower, upper, pelvis, hips), foot/hand/head names.
      - Motion tracking config (links, SMPL mapping, `joint_matches`, `smpl_pose_modifier`, visualization).
  - Interactions:
    - Used by all envs and simulators for consistency.
    - Validated by `IsaacGym.load_assets()` via assertions that DOF/body names match the asset.

- **`config/obs`, `config/rewards`, `config/domain_rand`, `config/terrain`, `config/opt`**
  - Responsibilities:
    - Structurally define what each environment observes (`obs`), how reward terms are computed (`rewards`), randomized properties (`domain_rand`), terrain generation parameters (`terrain`), and evaluation/logging/recording options (`opt`).
  - Interactions:
    - Their fields are accessed inside `LeggedRobotBase`, `LeggedRobotLocomotion`, `LeggedRobotMotionTracking`, Delta-A tasks, and the reward/termination/observation code.

### 3.3 Environment & Task Modules

- **`envs/base_task/base_task.py` (`BaseTask`)**
  - Responsibilities:
    - High-level RL environment interface built on `BaseSimulator`.
    - During `__init__`:
      - Instantiates simulator from `config.simulator._target_` via Hydra’s `get_class` and manual construction.
      - Configures headless mode and sets up simulator (`setup`).
      - Derives timing: `sim_dt`, `dt = control_decimation * sim_dt`, `max_episode_length`.
      - Sets env dimension parameters: `num_envs`, `dim_obs`, `dim_critic_obs`, `dim_actions` from robot config.
      - Calls `simulator.setup_terrain`, asset loading and env creation pipeline: `_load_assets()`, `_get_env_origins()`, `_create_envs()`, `get_dof_limits_properties()`, `_setup_robot_body_indices()`, `prepare_sim()`.
      - Optionally sets up viewer and debug visualization when not headless.
      - Initializes environment buffers: obs buffers, reward, reset, episode length, timeout, extras/logs.
    - Provides:
      - `reset_all()` — Reset all envs and return initial observation dict.
      - `step(actor_state)` — Not fully shown in snippets, but it coordinates simulation step, reward/termination, obs computation, and info dict.
      - Hooks `_init_buffers`, `_refresh_sim_tensors`, `_check_termination`, `_update_timeout_buf`, etc., to be overridden/extended by subclasses.
  - Interactions:
    - Strongly coupled to `BaseSimulator` (Isaac Gym / Isaac Sim / Genesis / MJLab implementations).
    - Extended by `LeggedRobotBase`, and then by locomotion and motion-tracking variants.

- **`envs/legged_base_task/legged_robot_base.py` (`LeggedRobotBase`)**
  - Responsibilities:
    - Extends `BaseTask` with legged humanoid specifics (control, kinematics, history, rewards).
    - In `__init__`:
      - Calls parent `BaseTask.__init__`, then `_domain_rand_config()`, `_prepare_reward_function()`.
      - Instantiates `HistoryHandler` and sets `is_evaluating` flag.
    - `_init_buffers`:
      - Extends base buffers with:
        - Base orientation (`base_quat`, `rpy`), velocities, gravity vector, forward vector.
        - PD control parameters (`p_gains`, `d_gains`), DOF default positions, torques, action buffers.
        - Contact-related data: feet indices, air time, last contacts.
        - Domain randomization buffers, average episode length, termination curriculum metrics.
      - Validates that PD gains are defined for all DOFs when using P/V control; otherwise raises.
    - Provides:
      - Domain randomization routines (`_domain_rand_config`, `_init_domain_rand_buffers`, episodic randomization hooks).
      - Reward computation hooks, history handling, and common step logic.
  - Interactions:
    - Used as the base class by locomotion and motion tracking envs.
    - Relies heavily on `config.robot`, `config.rewards`, `config.obs`, `config.domain_rand`.

- **`envs/locomotion/locomotion.py` (`LeggedRobotLocomotion`)**
  - Responsibilities:
    - Implements locomotion-specific behavior:
      - Gait phase management, command resampling (velocity/heading), arm swing targets, and gait parameters.
      - Extension of observation spaces with commands and gait-related features.
      - Terrain-aware locomotion metrics.
    - `_init_buffers`:
      - Adds command buffers (`commands` tensor), command ranges, per-env phase/gait buffers.
    - `_init_gait_params`:
      - Sets gait phase intervals, Von Mises distribution parameters, offsets, time buffers, and target arm poses.
    - `_setup_simulator_control`:
      - Writes locomotion commands into the simulator object for use in dynamics.
    - `_update_tasks_callback`:
      - Called each step before term/reward/obs; handles command resampling and heading tracking.
  - Interactions:
    - Uses `LeggedRobotBase` infrastructure for low-level robot control and reward logic.
    - Reads from `config.locomotion_command_ranges`, `config.rewards`, and `config.obs`.

- **`envs/motion_tracking/motion_tracking.py` (`LeggedRobotMotionTracking`)**
  - Responsibilities:
    - Implements motion tracking tasks where the humanoid follows reference motion trajectories.
    - In `__init__`:
      - Calls `LeggedRobotBase.__init__`, then sets up:
        - `_init_motion_lib()` — configure and load motion library for robot, including sampling initial states and storing motion dt/length.
        - `_init_motion_extend()` — extend skeleton with extra markers/bodies if defined (`extend_config`).
        - `_init_tracking_config()` — link mapping for motion tracking (which robot links are tracked, lower/upper body segmentation).
        - `_init_save_motion()` — optional logging of executed motion trajectories to disk.
      - Optionally sets up teleop via ROS 2 (`rclpy` and `Float64MultiArray` topic `vision_pro_data`).
      - Configures termination thresholds based on training or evaluation curricula.
    - Provides:
      - Teleop callback for VR markers.
      - Motion resampling and domain randomization for motion starting time and IDs.
      - Buffers for markers, reference positions, differences, VR keypoints, and curriculum state.
      - Overrides for `_reset_tasks_callback`, `_update_tasks_callback`, `_check_termination`, `_update_timeout_buf` to implement custom motion-based termination conditions (e.g., motion-far, motion-end).
  - Interactions:
    - Strongly coupled with `utils.motion_lib` (skeleton, motion_lib_robot).
    - Relies on `config.robot.motion` subtree extensively.

- **`envs/delta_a/*.py` (`DeltaA_ClosedLoop`, `DeltaA_OpenLoop`)**
  - Responsibilities (DeltaA_ClosedLoop shown):
    - Extends `LeggedRobotMotionTracking` to study Delta-A corrective policy behavior on top of a baseline motion-tracking policy.
    - `_init_buffers`:
      - Adds `with_delta_a_or_not`, `delta_a_scale` to support co-training “with/without Delta-A” and scaling of Delta-A.
    - `_episodic_domain_randomization`:
      - Re-randomizes `delta_a_scale` per episode when enabled.
    - `_compute_torques`:
      - Core Delta-A injection path:
        - Scales actions by `config.robot.control.action_scale`.
        - Optionally retrieves closed-loop policy action (`get_closed_loop_action_at_current_timestep`) and mixes with base action.
        - Applies randomization toggles (`cotrain_with_without_delta_a`, `rescale_delta_a`, action noise).
        - Supports gradient-search and fixed-point iteration for Delta-A when extra policies are loaded.
        - Implements ankle PR isolation when requested.
        - Converts to torques via PD/velocity/torque control modes, uses domain randomized torque perturbations, and torque clipping.
  - Interactions:
    - Consumes actor_state from algorithms that provide both `actions` and `actions_closed_loop`.
    - For research workflows, it is used together with `agents/delta_a/train_delta_a.PPODeltaA`.

### 3.4 Algorithm & Module Layer

- **`agents/base_algo/base_algo.py` (`BaseAlgo`)**
  - Responsibilities:
    - Defines the minimal interface for learning algorithms in this project:
      - Constructor takes `env: BaseTask`, `config`, and `device`.
      - Abstract methods: `setup()`, `learn()`, `load(path)`, `inference_model` property, `evaluate_policy()`, `save()`.
      - `env_step(self, actions, extra_info=None)` — thin wrapper over `env.step`.
    - Serves as the common type for `train_agent.py`/`eval_agent.py` and downstream algorithms.
  - Interactions:
    - All concrete algorithms (PPO, MPPI, DAgger, DeltaA variants, etc.) must subclass `BaseAlgo`.

- **`agents/ppo/ppo.py` (`PPO`)**
  - Responsibilities:
    - Implements PPO algorithm on top of `BaseAlgo`.
    - Constructor:
      - Stores device, env, config, log_dir.
      - Sets up TensorBoard writer, timing accumulators, reward/episode tracking buffers.
      - Calls `_init_config()` to read values from `self.config` and `self.env.config`.
      - Initializes evaluation callbacks and metric meters.
      - Calls `env.reset_all()` to warm start.
    - `_init_config`:
      - Extracts `num_envs`, `algo_obs_dim_dict`, `num_act` from `env.config.robot`.
      - Reads training hyperparameters: `save_interval`, `num_steps_per_env`, `load_optimizer`, `num_learning_iterations`, `init_at_random_ep_len`, and PPO-specific parameters (KL, LR, clip, epochs, minibatches, gamma, lam, value/entropy coefficients, max_grad_norm, etc.).
    - `setup`:
      - Creates actor/critic networks via `_setup_models_and_optimizer` using `PPOActor` and `PPOCritic`.
      - Initializes rollout storage via `_setup_storage`.
    - `_setup_storage`:
      - Uses `RolloutStorage` to register observation keys and scalar fields (`actions`, `rewards`, `dones`, `values`, `returns`, `advantages`).
    - `_rollout_step` (partially shown in output):
      - Loops for `num_steps_per_env`, performing actor/critic eval, stepping env, storing transitions, updating episode statistics, and computing returns/advantages using `_compute_returns` with GAE.
    - `_compute_returns`:
      - Computes returns and advantages using last value estimates, `gamma`, `lam`, and `dones`.
    - `learn` (not shown but typical):
      - Iterates over rollouts, updates networks by optimizing PPO loss using minibatches from `RolloutStorage`.
    - `evaluate_policy`:
      - Performs evaluation episodes with logging and optional callback invocation.
  - Interactions:
    - Heavily uses `agents/modules/ppo_modules` and `agents/modules/data_utils.RolloutStorage`.
    - Env step uses `env.step(actor_state)` with dict-based actor_state, expecting envs to interpret `actions` appropriately (e.g., as PD targets).

- **`agents/delta_a/train_delta_a.py` (`PPODeltaA`)**
  - Responsibilities:
    - Extends `PPO` to train a Delta-A policy on top of a pre-trained baseline policy:
      - Loads baseline policy from `config.policy_checkpoint`, reconstructs its config (including `eval_overrides`).
      - Calls `pre_process_config` on baseline config.
      - Instantiates `loaded_policy: BaseAlgo` with original env, sets its observation dimensions, and loads weights.
      - Freezes baseline policy’s actor parameters and obtains an evaluation policy handle (`eval_policy`).
    - Overrides `_rollout_step`:
      - For each step:
        - Calls standard PPO actor/critic to compute RL actions.
        - Queries `loaded_policy.eval_policy` with `obs_dict['closed_loop_actor_obs']` to get `actions_closed_loop`.
        - Packs `actions` and `actions_closed_loop` into actor_state, and passes to `env.step`.
      - Computes returns/advantages based on rewards/dones like PPO.
    - Provides `_pre_eval_env_step`:
      - For evaluation, computes both `actions` and `actions_closed_loop` before env step.
  - Interactions:
    - Designed to work with DeltaA environments that interpret `actions` and `actions_closed_loop` separately.
    - Interoperates with same `RolloutStorage` and PPO training loop.

- **`agents/modules/*`**
  - Responsibilities:
    - `ppo_modules.py`:
      - Defines `PPOActor`, `PPOCritic` networks using modular components and `module_config_dict`.
      - Provides `reset` and `evaluate` methods used in PPO training loops.
    - `data_utils.py`:
      - Implements `RolloutStorage` that stores multi-key dictionary-like experience for batched environments and multi-step rollouts.
      - Provides `register_key`, `update_key`, `batch_update_data`, `query_key`, `increment_step`, etc.
    - `world_models.py`, `encoder_modules.py`, `modules.py`:
      - Provide higher-level networks for world modeling, representation learning, encoders, etc., used by some specialized agents.
  - Interactions:
    - Shared across PPO, DeltaA, and potentially other agent types.

- **`agents/callbacks/*`**
  - Responsibilities:
    - `base_callback.py`:
      - Defines `RL_EvalCallback` base and utilities for evaluation-time hooks (pre/post env step, logging, analysis).
    - Other callbacks (e.g., `analysis_plot_locomotion.py`, `analysis_plot_motion_tracking.py`, `analysis_plot_force_estimator.py`) implement domain-specific analysis and plotting.
  - Interactions:
    - Used by algorithms to plug in additional logging/plotting logic without entangling training loops.

### 3.5 Simulator Modules

- **`simulator/base_simulator/base_simulator.py` (`BaseSimulator`)**
  - Responsibilities:
    - Defines abstract interface for all simulator backends:
      - `setup`, `setup_terrain`, `load_assets`, `create_envs`, `prepare_sim`, `refresh_sim_tensors`.
      - DOF and body property retrieval (`get_dof_limits_properties`, `find_rigid_body_indice`).
      - Control actions (`apply_torques_at_dof`), state setters (`set_actor_root_state_tensor`, `set_dof_state_tensor`).
      - Simulation stepping (`simulate_at_each_physics_step`).
      - Viewer setup/render (`setup_viewer`, `render`).
    - Holds `config`, `sim_device`, headless flag, and typed tensor placeholders for rigid body states.
  - Interactions:
    - Implemented by each backend (Isaac Gym, Isaac Sim, Genesis, MJLab).
    - Used by BaseTask and environment subclasses.

- **`simulator/isaacgym/isaacgym.py` (`IsaacGym`)**
  - Responsibilities:
    - Implementation of `BaseSimulator` for NVIDIA Isaac Gym.
    - `setup`:
      - Configures `SimParams` from `simulator_config` (`fps`, `substeps`, `physx` parameters).
      - Creates `gym` and `sim`, sets `sim_dt`, device selection, GPU pipeline mode, graphics device ID (with headless support).
    - `setup_terrain`:
      - Depending on `mesh_type`, uses `envs.env_utils.terrain.Terrain` to generate heightfields/trimesh or a plane.
      - Adds appropriate terrain to the sim.
    - `load_assets`:
      - Sets up robot asset from URDF with `gym.load_asset`, reading options from `robot_config.asset`.
      - Asserts DOF and body names from the URDF match the robot config.
    - Additional responsibilities (not fully shown, but implied):
      - `create_envs`, `prepare_sim`, state accessors for positions/velocities, DOF and body-level state.
      - Support for rendering, cameras, and debug visualization.
  - Interactions:
    - Used by `BaseTask`/`LeggedRobotBase` via `config.simulator._target_`.
    - Reads robot config information for DOF layout and PD control.

- **Other simulators: `simulator/isaacsim/*`, `simulator/genesis/*`, `simulator/mjlab/*`**
  - Responsibilities:
    - Provide backend-specific environment setup, asset loading, and stepping compatible with `BaseSimulator`.
    - `isaacsim` includes articulation config, events, and view controllers.
    - `genesis` and `mjlab` provide alternatives for research under different engines or internal labs.
  - Interactions:
    - Selectable via `config/simulator/*.yaml` and `config.simulator._target_`.

### 3.6 Utilities & Motion Libraries

- **`utils/config_utils.py`**
  - Responsibilities:
    - Registers OmegaConf resolvers (`eval`, `if`, `eq`, `sqrt`, `sum`, `ceil`, `int`, `len`, `sum_list`) with try/except to avoid duplicate registration warnings.
  - Interactions:
    - Imported in `train_agent.py`/`eval_agent.py` prior to Hydra instantiation to ensure resolvers are available.

- **`utils/common.py`**
  - Responsibilities:
    - Utilities for argument parsing conflict resolution, colored terminal output, timestamp generation.
    - `seeding(seed, torch_deterministic)` to configure deterministic/non-deterministic behavior across Python/NumPy/Torch and CUDA, plus `CUBLAS_WORKSPACE_CONFIG`.
  - Interactions:
    - Recommended to be called from training scripts for reproducibility.

- **`utils/helpers.py`, `utils/inference_helpers.py`, `utils/logging.py`, `utils/torch_utils.py`, `utils/math.py`, `utils/terrain.py`**
  - Responsibilities:
    - `helpers.pre_process_config` — cleans up and enriches config before env/algo instantiation (e.g., obs dims, derived fields, device specifics).
    - `inference_helpers` — exports trained policies to TorchScript and ONNX (including force-estimation variants).
    - `logging.HydraLoggerBridge` — forwards Python logging to Loguru, integrated in `train_agent.py`/`eval_agent.py`.
    - `torch_utils` — common tensor operations, randomization utilities.
    - `math`, `terrain` — math helpers, terrain utilities complementary to env-level terrain.
  - Interactions:
    - Widely used across envs, simulators, and agents.

- **`utils/motion_lib/*`**
  - Responsibilities:
    - `skeleton.py` — defines `SkeletonTree` representation.
    - `motion_lib_robot.py` — motion library for robot-specific motions, with methods to load and query motion states.
    - `torch_humanoid_batch.py` — batched humanoid motion operations.
  - Interactions:
    - Used by motion-tracking envs and Delta-A tasks.

### 3.7 Data Flow Summary

- Config flow:
  - Hydra loads `config/base.yaml` + defaults which include `config/base/structure.yaml` and specific `algo/env/robot/terrain` configs.
  - `pre_process_config` mutates/augments this config (e.g., sets `log_task_name`, ensures `robot.policy_obs_dim`/`critic_obs_dim` are consistent).
- Runtime flow:
  - `train_agent.py`/`eval_agent.py`:
    - Instantiate env via `config.env` => `BaseTask` or `LeggedRobotBase` variant, which instantiates `BaseSimulator` and sets up simulation.
    - Instantiate algo via `config.algo` => `BaseAlgo` subclass (PPO, PPODeltaA, etc.).
  - During training:
    - Algo loops: actor/critic -> env.step -> compute rewards/term/obs -> record in storage -> PPO updates.
  - Observations:
    - Env composes observation dict(s) keyed by e.g. `actor_obs`, `critic_obs`, `closed_loop_actor_obs`, etc., with dimensions described in robot/obs configs.
  - Actions:
    - Algo passes `actor_state` dict to env; env dispatches to control logic (`_compute_torques`) that uses config, domain-rand, and motion-tracking specifics.

---

## 4. Technical Constraints

- **Conda environment usage**
  - All code execution, debugging, unit tests, integration tests, simulation runs, and training/evaluation scripts must be run inside a Conda environment specifically for mjlab simulator:
    `conda activate ASAP_mjlab`

- **Execution assumptions**
  - Simulators:
    - Some simulators (IsaacGym, IsaacSim, Genesis, MJLab) have specific system requirements and may not be available in all environments.
    - Agents should not attempt to modify simulator installation logic; instead, rely on existing configs and expect the environment to provide the necessary SDKs.
  - Devices:
    - Default device selection is `"cuda:0"` when available, falling back to `"cpu"`. This may be overridden by `config.device`.
    - Algorithms and envs assume batched GPU tensors and rely on CUDA performance.
  - Logging:
    - Logging is handled via Loguru plus standard logging bridged into Hydra output directories. Do not rewire logging globally; instead, plug into existing utilities (`HydraLoggerBridge`, W&B flags).

---

## 5. Coding Preference

The following conventions are **project-wide guidelines** that any AI code agent must follow when modifying or extending this repository.

- **Prefer minimal, surgical changes**
  - Keep modifications small, localized, and directly tied to the feature/bug you are addressing.
  - Avoid large-scale refactors across modules unless explicitly necessary and agreed upon (e.g., breaking changes to a central interface).
  - When adjusting behavior, favor adding clearly-scoped hooks or helper functions over rewriting core pipelines.

- **Maintain consistent tensor/obs/action shapes**
  - All observation, action, and internal tensors must obey clearly defined, consistent shapes.
  - Observation dicts should match `algo_obs_dim_dict`, `policy_obs_dim`, and `critic_obs_dim` in robot/env configs.
  - Shape mismatches (e.g., between env outputs and `PPOActor` inputs) must be treated as hard errors; add asserts where needed.
  - When adding new observation keys, update both config and env/algorithm code to keep shapes synchronized.

- **Add docstrings to all public functions**
  - Any function/method that is part of a public API (e.g., in `BaseTask`, `BaseAlgo`, environment classes, simulator interfaces, core utilities) must have a docstring.
  - Document:
    - Inputs (types, shapes, expected ranges).
    - Outputs (types, shapes, semantics).
    - Side effects (state updates, logging, file IO).
    - Failure modes and assumptions (e.g., device, dtype, valid ranges).
  - For complex RL pipelines (rollout, storage, export), ensure docstrings explain the data flow clearly.

- **Use precise, descriptive names**
  - Choose names that convey intent: `resample_motion_times`, `get_closed_loop_action_at_current_timestep`, `terminate_when_motion_far_threshold` are good patterns.
  - Avoid cryptic abbreviations (`x`, `y`, `tmp`) for persistent variables, especially in environment/algorithm code.
  - Preserve existing naming patterns for related concepts (e.g., `*_buf` for buffers, `*_ids` for indices, `*_config` for config objects).

- **Avoid deep nesting**
  - Use guard clauses and early returns to reduce indentation depth in functions with branching logic.
  - For complex RL step functions or torque computation functions, break out nested logic into small helpers.
  - Replace deeply nested `if/else` ladders with clear helper methods or lookup tables where appropriate.

- **One function does one job**
  - Keep methods focused: a single function should perform one conceptual task (e.g., “compute rewards”, “update domain-rand buffers”, “export ONNX model”), not multiple unrelated steps.
  - For long functions that are already in the codebase, new logic should be added as separate helpers rather than further expanding their responsibilities.

- **Organize code into clear modules (dataset/model/trainer/utils)**
  - Although this project is RL/env-centric rather than dataset-centric, adhere to modular separation:
    - Env definitions and control logic in `envs/`.
    - Algorithm logic and models in `agents/` and `agents/modules/`.
    - Simulator-specific details in `simulator/`.
    - Reusable math & configuration utilities in `utils/`.
    - Configuration in `config/`.
    - Assets (motions, robots) in `data/`.
  - New features should be slotted into the appropriate vertical slice instead of introducing cross-cutting modules.

- **Never reuse variables for different meanings**
  - A variable name must retain the same semantic meaning for its entire lifetime within a scope.
  - Do not reassign a tensor to something structurally unrelated (e.g., reusing `actions` for both raw and scaled actions; prefer `actions_scaled`).
  - When code historically reuses variables in this way, prefer to clean it up when touching that code path.

- **Log important events, not everything**
  - Use Loguru and existing logging patterns to log:
    - Start/end of major phases (training iterations, eval runs, export steps).
    - Key metrics (rewards, episode lengths, gait/motion statistics, curriculum thresholds).
    - Configuration decisions (simulator type, device, environment selection).
  - Avoid excessive debug prints in the main loop; use debug-level logs guarded by configuration flags if necessary.
  - Remove or gate existing `print` statements when they become noisy, preferring `logger.debug/info`.

- **Assert aggressively for invariants**
  - Use `assert` statements and explicit checks to:
    - Validate tensor shapes, dtypes, and devices (`cuda` vs `cpu`).
    - Ensure DOF/bodynames in robot configs match simulator asset introspection (already done in `IsaacGym.load_assets`).
    - Ensure observation dicts contain the expected keys before passing to policies.
  - When invariants are violated, raise clear, descriptive exceptions (e.g., `ValueError`, `RuntimeError`) explaining the mismatch.

- **Separate configuration from logic**
  - All hyperparameters, environment parameters, paths, and experiment settings must live in configs (`config/`) or CLI arguments, not hardcoded inside logic.
  - When introducing a new tunable parameter (e.g., a reward scale, a domain-rand toggle, a new motion filter), add it to the appropriate config YAML and read it from `config` in the code.
  - Avoid embedding file paths directly in code; instead, derive them from config fields (e.g., `config.wandb.wandb_dir`, `config.env.config.save_rendering_dir`, `config.robot.asset.asset_root`).

- **Adopt a meaningful folder structure**
  - Maintain and extend the existing structure:
    - Keep new envs under `envs/<task_name>/`.
    - Keep new algorithms under `agents/<algo_name>/`.
    - Keep new simulator backends under `simulator/<backend_name>/`.
    - Create new config groups under `config/<group>/` instead of ad hoc YAMLs.
  - When adding completely new capabilities (e.g., new motion types or robots), follow the existing patterns (`data/motions`, `data/robots`, `config/robot/<robot_name>`, `config/env/<task>_<robot>.yaml`).

- **Fail loudly, not silently**
  - When encountering inconsistent configs (missing `???` values, mismatched DOF lists, invalid `sim_type`), raise explicit exceptions instead of silently defaulting.
  - For unsupported simulator/features, raise `NotImplementedError` or descriptive `ValueError`.
  - Use logs plus exceptions when failing at runtime to help future debugging.

- **Use type hints wherever practical**
  - Add type hints to function signatures, especially in:
    - Public APIs (`BaseTask`, `BaseAlgo`, `RolloutStorage`, simulators, helpers).
    - New code paths in envs and agents.
  - For tensors, use `torch.Tensor` / `Tensor` with shape hints in docstrings.
  - Avoid overcomplicating types; prefer readable hints over very deep generic types, but annotate dicts and lists with key/value types when helpful (e.g., `Dict[str, torch.Tensor]`, `List[int]`).

---

This document is intended as the canonical guidance for future AI code agents working on this repository. When in doubt, keep changes minimal, respect existing abstractions (`BaseTask`, `BaseAlgo`, `BaseSimulator`, Hydra configs), enforce shape and config invariants, and surface issues via clear logging and explicit failures.

