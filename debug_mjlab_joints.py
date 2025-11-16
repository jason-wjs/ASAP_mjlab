#!/usr/bin/env python3
"""
诊断 MJLab 关节映射问题的脚本
检查：
1. 关节顺序是否匹配
2. 关节轴方向是否正确
3. qpos/qvel 地址计算是否正确
"""

import mujoco
import numpy as np
from pathlib import Path

# G1 的关节顺序（来自 config）
EXPECTED_JOINTS = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
    'left_elbow_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
    'right_elbow_joint'
]

# 加载 MJCF 模型
xml_path = Path("/home/wujs/Projects/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.xml")
if not xml_path.exists():
    print(f"❌ 找不到 MJCF 文件: {xml_path}")
    exit(1)

model = mujoco.MjModel.from_xml_path(str(xml_path))
print(f"✅ 成功加载模型: {xml_path}")
print(f"   模型总关节数: {model.njnt}")
print(f"   模型总自由度: {model.nv}")
print()

# 打印所有关节信息
print("=" * 80)
print("MuJoCo 模型中的所有关节：")
print("=" * 80)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_type = model.jnt_type[i]
    qposadr = model.jnt_qposadr[i]
    dofadr = model.jnt_dofadr[i]
    axis = model.jnt_axis[i]
    range_min, range_max = model.jnt_range[i]
    
    type_name = ['FREE', 'BALL', 'SLIDE', 'HINGE'][jnt_type]
    
    print(f"[{i:2d}] {name:30s} | Type: {type_name:6s} | qpos_adr: {qposadr:3d} | dof_adr: {dofadr:3d} | axis: {axis} | range: [{range_min:7.3f}, {range_max:7.3f}]")

print()
print("=" * 80)
print("检查 HumanoidVerse 配置中的关节映射：")
print("=" * 80)

# 检查每个期望的关节
missing_joints = []
mapped_joints = []
axis_warnings = []

for idx, expected_name in enumerate(EXPECTED_JOINTS):
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, expected_name)
        qposadr = model.jnt_qposadr[jid]
        dofadr = model.jnt_dofadr[jid]
        axis = model.jnt_axis[jid]
        jnt_type = model.jnt_type[jid]
        
        mapped_joints.append({
            'config_idx': idx,
            'config_name': expected_name,
            'mj_id': jid,
            'qposadr': qposadr,
            'dofadr': dofadr,
            'axis': axis
        })
        
        # 检查关节轴方向
        expected_axes = {
            'hip_pitch': [0, 1, 0],
            'hip_roll': [1, 0, 0],
            'hip_yaw': [0, 0, 1],
            'knee': [0, 1, 0],
            'ankle_pitch': [0, 1, 0],
            'ankle_roll': [1, 0, 0],
            'waist_yaw': [0, 0, 1],
            'waist_roll': [1, 0, 0],
            'waist_pitch': [0, 1, 0],
            'shoulder_pitch': [0, 1, 0],
            'shoulder_roll': [1, 0, 0],
            'shoulder_yaw': [0, 0, 1],
            'elbow': [0, 1, 0],
        }
        
        for key, expected_axis in expected_axes.items():
            if key in expected_name:
                if not np.allclose(axis, expected_axis, atol=0.1):
                    axis_warnings.append(f"  ⚠️  {expected_name}: 期望轴 {expected_axis}, 实际轴 {axis.tolist()}")
        
        status = "✅"
        print(f"{status} [{idx:2d}] {expected_name:30s} -> MJ_ID: {jid:2d} | qpos: {qposadr:3d} | dof: {dofadr:3d} | axis: {axis.tolist()}")
        
    except ValueError:
        missing_joints.append(expected_name)
        print(f"❌ [{idx:2d}] {expected_name:30s} -> 未找到！")

print()
print("=" * 80)
print("诊断总结：")
print("=" * 80)
print(f"✅ 成功映射的关节: {len(mapped_joints)}/{len(EXPECTED_JOINTS)}")
print(f"❌ 缺失的关节: {len(missing_joints)}")

if missing_joints:
    print("\n缺失的关节列表：")
    for name in missing_joints:
        print(f"  - {name}")

if axis_warnings:
    print("\n⚠️  关节轴方向警告：")
    for warning in axis_warnings:
        print(warning)

# 检查 qpos/qvel 地址的连续性
print()
print("=" * 80)
print("检查 DOF 地址的连续性：")
print("=" * 80)
dof_addrs = [j['dofadr'] for j in mapped_joints]
sorted_addrs = sorted(dof_addrs)
print(f"配置顺序的 DOF 地址: {dof_addrs}")
print(f"排序后的 DOF 地址: {sorted_addrs}")

if dof_addrs == sorted_addrs:
    print("✅ DOF 地址是连续且有序的")
else:
    print("⚠️  DOF 地址不连续或顺序不对！这可能导致关节数据错乱")
    print("\n详细对比：")
    for idx, (config_addr, sorted_addr) in enumerate(zip(dof_addrs, sorted_addrs)):
        if config_addr != sorted_addr:
            print(f"  位置 {idx}: 配置顺序 DOF={config_addr}, 应该是 DOF={sorted_addr}")

print()
print("=" * 80)
print("生成 MJLab 调试命令：")
print("=" * 80)
print("""
# 在训练时添加详细日志：
export LOGURU_LEVEL=DEBUG

# 单环境测试（更容易调试）：
python humanoidverse/train_agent.py \\
    +simulator=mjlab \\
    +exp=motion_tracking \\
    num_envs=1 \\
    headless=False \\
    simulator.config.sim.viewer.mode=viser

# 检查 NaN dump：
uv run viz-nan /tmp/mjlab/nan_dumps/nan_dump_latest.npz
""")
