#!/usr/bin/env python3
"""
分析 MJLab NaN dump 文件，定位问题根源
"""
import numpy as np
import sys
from pathlib import Path

dump_path = Path("/tmp/mjlab/nan_dumps/nan_dump_latest.npz")
if not dump_path.exists():
    print(f"❌ 找不到 NaN dump 文件: {dump_path}")
    sys.exit(1)

data = np.load(dump_path, allow_pickle=True)

print("=" * 80)
print(f"加载 NaN Dump 文件: {dump_path}")
print("=" * 80)

print("\n可用的键：")
step_keys = []
for key in sorted(data.keys()):
    val = data[key]
    if key.startswith('states_step_'):
        step_keys.append(key)
    if isinstance(val, np.ndarray):
        print(f"  {key:30s}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"  {key:30s}: {type(val)}")

print(f"\n找到 {len(step_keys)} 个时间步的数据")

print("\n=" * 80)
print("分析 NaN 出现的位置")
print("=" * 80)

# G1 关节名称
joint_names = [
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 
    'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
    'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 
    'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
    'waist_yaw', 'waist_roll', 'waist_pitch',
    'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow'
]

def analyze_array(name, arr, step_idx=-1):
    """分析数组中的 NaN/Inf"""
    if arr is None:
        return
    
    if step_idx >= 0:
        arr = arr[step_idx]
    
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    
    if has_nan or has_inf:
        print(f"\n🔴 {name} [step {step_idx}]:")
        if has_nan:
            nan_indices = np.where(np.isnan(arr))
            print(f"   NaN 位置: {nan_indices}")
            if len(arr.shape) == 1 and len(arr) == 23:  # 关节数据
                for idx in nan_indices[0]:
                    print(f"     [{idx:2d}] {joint_names[idx]:25s} = NaN")
        if has_inf:
            inf_indices = np.where(np.isinf(arr))
            print(f"   Inf 位置: {inf_indices}")
    
    return has_nan or has_inf

# 检查最后几步的数据
print(f"\n记录的步数: {len(step_keys)}")

if len(step_keys) > 0:
    # 分析最后5步
    for key in step_keys[-5:]:
        step_num = int(key.split('_')[-1])
        state = data[key][0]  # shape (1, 59) -> (59,)
        
        print(f"\n{'='*80}")
        print(f"Step {step_num} - State shape: {state.shape}")
        print(f"{'='*80}")
        
        # MuJoCo state 格式: [qpos (30), qvel (29)]
        # qpos: [x, y, z, qw, qx, qy, qz, joint1, joint2, ...]  (7 + 23 = 30)
        # qvel: [vx, vy, vz, wx, wy, wz, joint_vel1, ...]       (6 + 23 = 29)
        
        qpos = state[:30]
        qvel = state[30:59]
        
        # 根刚体状态
        base_pos = qpos[:3]
        base_quat_wxyz = qpos[3:7]
        print(f"\n根刚体位置: {base_pos}")
        print(f"根刚体四元数 (WXYZ): {base_quat_wxyz}")
        
        # 关节位置
        joint_pos = qpos[7:]
        print(f"\n关节位置 (qpos) [shape={joint_pos.shape}]:")
        for i, (name, val) in enumerate(zip(joint_names, joint_pos)):
            status = "🔴 NaN" if np.isnan(val) else "🟡 Inf" if np.isinf(val) else "✅"
            # 添加角度范围检查
            deg_val = np.rad2deg(val) if not (np.isnan(val) or np.isinf(val)) else 0
            print(f"  [{i:2d}] {name:25s}: {val:10.4f} rad ({deg_val:7.2f}°)  {status}")
        
        # 根刚体速度
        base_lin_vel = qvel[:3]
        base_ang_vel = qvel[3:6]
        joint_vel = qvel[6:]
        
        print(f"\n根刚体线速度: {base_lin_vel}")
        print(f"根刚体角速度: {base_ang_vel}")
        
        has_nan_vel = np.isnan(joint_vel).any()
        has_inf_vel = np.isinf(joint_vel).any()
        if has_nan_vel or has_inf_vel:
            print(f"\n关节速度 (qvel) 异常:")
            for i, (name, val) in enumerate(zip(joint_names, joint_vel)):
                if np.isnan(val) or np.isinf(val):
                    status = "🔴 NaN" if np.isnan(val) else "🟡 Inf"
                    print(f"  [{i:2d}] {name:25s}: {val:10.4f}  {status}")

print("\n=" * 80)
print("关键发现总结")
print("=" * 80)

# 检查是否有特定关节一直有问题
if len(step_keys) > 0:
    all_joint_pos = []
    all_joint_vel = []
    
    for key in step_keys:
        state = data[key][0]
        qpos = state[:30]
        qvel = state[30:59]
        all_joint_pos.append(qpos[7:])
        all_joint_vel.append(qvel[6:])
    
    all_joint_pos = np.array(all_joint_pos)  # [steps, 23]
    all_joint_vel = np.array(all_joint_vel)
    
    nan_count_per_joint = np.isnan(all_joint_pos).sum(axis=0)
    print("\n每个关节(位置)出现 NaN 的次数：")
    for i, (name, count) in enumerate(zip(joint_names, nan_count_per_joint)):
        if count > 0:
            print(f"  🔴 [{i:2d}] {name:25s}: {count}/{len(all_joint_pos)} 步")
    
    # 检查关节范围
    print("\n关节位置范围检查（最后一步）：")
    last_joint_pos = all_joint_pos[-1]
    joint_limits = {
        'left_hip_pitch': (-2.5307, 2.8798),
        'left_hip_roll': (-0.5236, 2.9671),
        'left_hip_yaw': (-2.7576, 2.7576),
        'left_knee': (-0.087267, 2.8798),
        'left_ankle_pitch': (-0.87267, 0.5236),
        'left_ankle_roll': (-0.2618, 0.2618),
        'right_hip_pitch': (-2.5307, 2.8798),
        'right_hip_roll': (-2.9671, 0.5236),
        'right_hip_yaw': (-2.7576, 2.7576),
        'right_knee': (-0.087267, 2.8798),
        'right_ankle_pitch': (-0.87267, 0.5236),
        'right_ankle_roll': (-0.2618, 0.2618),
    }
    
    for i, (name, val) in enumerate(zip(joint_names, last_joint_pos)):
        if name in joint_limits:
            min_lim, max_lim = joint_limits[name]
            if not (np.isnan(val) or np.isinf(val)):
                if val < min_lim or val > max_lim:
                    print(f"  ⚠️  [{i:2d}] {name:25s}: {val:7.3f} 超出范围 [{min_lim:7.3f}, {max_lim:7.3f}]")

print("\n=" * 80)
print("建议的调试步骤")
print("=" * 80)
print("""
1. 检查初始姿态是否合理：
   - 关节角度是否在限制范围内
   - 初始速度是否为零

2. 检查动作缩放：
   - action_scale = 0.25 可能太大？
   - 尝试减小到 0.1 测试

3. 检查 PD 增益：
   - 是否有关节的 kp/kd 过大导致不稳定

4. 检查力矩限制：
   - 是否正确应用了力矩裁剪

5. 单步调试：
   python humanoidverse/train_agent.py \\
       +simulator=mjlab \\
       +exp=motion_tracking \\
       num_envs=1 \\
       robot.control.action_scale=0.1 \\
       headless=False
""")
