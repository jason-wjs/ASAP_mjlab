#!/usr/bin/env python3
"""
诊断关节运动方向问题
对比 IsaacGym URDF 和 MuJoCo MJCF 的关节轴定义
"""

import mujoco
import numpy as np
from pathlib import Path

# 加载 G1 的 MJCF 模型
xml_path = Path("/home/wujs/Projects/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.xml")
model = mujoco.MjModel.from_xml_path(str(xml_path))

print("=" * 80)
print("G1 机器人关节轴方向详细分析")
print("=" * 80)

# 重点关注下肢关节
critical_joints = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint', 
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
]

print("\n下肢关节轴方向和范围：")
print("-" * 80)

for joint_name in critical_joints:
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        axis = model.jnt_axis[jid]
        range_min, range_max = model.jnt_range[jid]
        pos = model.jnt_pos[jid]  # 关节在父体坐标系中的位置
        
        print(f"\n{joint_name}:")
        print(f"  轴方向: {axis}  (parent body frame)")
        print(f"  范围: [{range_min:7.3f}, {range_max:7.3f}] rad")
        print(f"  范围: [{np.rad2deg(range_min):7.2f}°, {np.rad2deg(range_max):7.2f}°]")
        print(f"  位置: {pos}")
        
        # 分析轴方向
        if 'hip_pitch' in joint_name:
            expected = [0, 1, 0]
            if not np.allclose(axis, expected):
                print(f"  ⚠️  警告: 期望 Y 轴 {expected}, 实际 {axis.tolist()}")
        elif 'hip_roll' in joint_name:
            expected = [1, 0, 0]
            if not np.allclose(axis, expected):
                print(f"  ⚠️  警告: 期望 X 轴 {expected}, 实际 {axis.tolist()}")
        elif 'hip_yaw' in joint_name:
            expected = [0, 0, 1]
            if not np.allclose(axis, expected):
                print(f"  ⚠️  警告: 期望 Z 轴 {expected}, 实际 {axis.tolist()}")
        elif 'knee' in joint_name:
            expected = [0, 1, 0]
            if not np.allclose(axis, expected):
                print(f"  ⚠️  警告: 期望 Y 轴 {expected}, 实际 {axis.tolist()}")
        elif 'ankle_pitch' in joint_name:
            expected = [0, 1, 0]
            if not np.allclose(axis, expected):
                print(f"  ⚠️  警告: 期望 Y 轴 {expected}, 实际 {axis.tolist()}")
        elif 'ankle_roll' in joint_name:
            expected = [1, 0, 0]
            if not np.allclose(axis, expected):
                print(f"  ⚠️  警告: 期望 X 轴 {expected}, 实际 {axis.tolist()}")
                
    except Exception as e:
        print(f"❌ {joint_name}: {e}")

print("\n" + "=" * 80)
print("关键观察 - 左右腿对称性检查")
print("=" * 80)

# 检查左右腿的关节限制是否对称
left_right_pairs = [
    ('left_hip_pitch_joint', 'right_hip_pitch_joint'),
    ('left_hip_roll_joint', 'right_hip_roll_joint'),
    ('left_hip_yaw_joint', 'right_hip_yaw_joint'),
    ('left_knee_joint', 'right_knee_joint'),
    ('left_ankle_pitch_joint', 'right_ankle_pitch_joint'),
    ('left_ankle_roll_joint', 'right_ankle_roll_joint'),
]

for left_name, right_name in left_right_pairs:
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left_name)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, right_name)
    
    left_axis = model.jnt_axis[left_id]
    right_axis = model.jnt_axis[right_id]
    
    left_range = model.jnt_range[left_id]
    right_range = model.jnt_range[right_id]
    
    print(f"\n{left_name.replace('_joint', '')} vs {right_name.replace('_joint', '')}:")
    print(f"  左轴: {left_axis}  |  右轴: {right_axis}")
    
    # 对于 roll 轴，左右应该相反或范围相反
    if 'roll' in left_name:
        # roll 轴方向应该相同，但范围可能相反
        print(f"  左范围: [{left_range[0]:6.2f}, {left_range[1]:6.2f}]")
        print(f"  右范围: [{right_range[0]:6.2f}, {right_range[1]:6.2f}]")
        if not (np.isclose(left_range[0], -right_range[1], atol=0.1) and 
                np.isclose(left_range[1], -right_range[0], atol=0.1)):
            print(f"  ⚠️  roll 关节范围不对称！")
    else:
        # pitch 和 yaw 应该完全相同
        if not np.allclose(left_axis, right_axis):
            print(f"  ⚠️  轴方向不一致！")
        if not np.allclose(left_range, right_range):
            print(f"  ⚠️  范围不一致: 左 {left_range} vs 右 {right_range}")

print("\n" + "=" * 80)
print("测试：应用小角度扰动，检查运动方向")
print("=" * 80)

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# 保存初始状态
qpos_init = data.qpos.copy()

test_cases = [
    ('left_hip_pitch_joint', 0.1, '左髋 pitch +0.1 rad (向前)'),
    ('left_hip_roll_joint', 0.1, '左髋 roll +0.1 rad (外展)'),
    ('left_knee_joint', 0.3, '左膝 +0.3 rad (弯曲)'),
    ('left_ankle_pitch_joint', -0.1, '左踝 pitch -0.1 rad (背屈)'),
]

for joint_name, delta, description in test_cases:
    # 重置
    data.qpos[:] = qpos_init
    
    # 应用扰动
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[jid]
    data.qpos[qpos_addr] += delta
    
    # 前向计算
    mujoco.mj_forward(model, data)
    
    # 获取末端位置变化（以脚踝为例）
    ankle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
    ankle_pos = data.xpos[ankle_body_id]
    
    print(f"\n{description}:")
    print(f"  脚踝世界坐标: {ankle_pos}")
