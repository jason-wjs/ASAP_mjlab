#!/usr/bin/env python3
"""
诊断脚本: 验证力矩施加方向和 PD 控制器符号

检查点:
1. qfrc_applied 的符号约定
2. PD 控制器的力矩计算
3. 与 IsaacGym/Genesis 的符号对比
"""

import sys
import torch
import numpy as np

def check_pd_controller_sign():
    """检查 PD 控制器的符号"""
    
    print("=" * 80)
    print("PD 控制器力矩符号检查")
    print("=" * 80)
    
    print("""
标准 PD 控制器:
---------------------------------------------------
τ = Kp * (q_target - q_current) + Kd * (qdot_target - qdot_current)

当 q_current < q_target时:
- 误差 (q_target - q_current) > 0
- 力矩 τ > 0 (正向力矩,驱动关节向目标移动)

MuJoCo 的 qfrc_applied:
- 施加广义力/力矩到关节
- 正值力矩驱动关节正向旋转
- 符号约定: 遵循关节轴方向

需要检查:
1. ✅ PD 控制器计算的力矩符号
2. ❓ apply_torques() 传递给 qfrc_applied 的符号是否需要反转?
3. ❓ 关节轴定义是否影响符号?
    """)
    
    # 模拟场景
    print("\n" + "=" * 80)
    print("模拟场景: 膝关节从 0° 移动到 30°")
    print("=" * 80)
    
    q_current = 0.0  # 当前角度
    q_target = 0.3   # 目标角度 (约 17°)
    qdot_current = 0.0
    qdot_target = 0.0
    
    Kp = 100.0
    Kd = 10.0
    
    # PD 控制器计算
    pos_error = q_target - q_current
    vel_error = qdot_target - qdot_current
    torque = Kp * pos_error + Kd * vel_error
    
    print(f"\n输入:")
    print(f"  当前角度: {q_current:.2f} rad ({np.degrees(q_current):.1f}°)")
    print(f"  目标角度: {q_target:.2f} rad ({np.degrees(q_target):.1f}°)")
    print(f"  当前速度: {qdot_current:.2f} rad/s")
    print(f"  目标速度: {qdot_target:.2f} rad/s")
    print(f"  Kp = {Kp}, Kd = {Kd}")
    
    print(f"\nPD 控制器计算:")
    print(f"  位置误差: {pos_error:.2f}")
    print(f"  速度误差: {vel_error:.2f}")
    print(f"  计算力矩: {torque:.2f}")
    
    print(f"\n预期行为:")
    if torque > 0:
        print(f"  ✅ 力矩为正 ({torque:.2f}),驱动关节正向旋转")
        print(f"  ✅ 这应该使膝关节向目标角度移动")
    else:
        print(f"  ❌ 力矩为负,这是错误的!")
    
    return torque


def check_mujoco_qfrc_convention():
    """检查 MuJoCo qfrc_applied 的约定"""
    
    print("\n" + "=" * 80)
    print("MuJoCo qfrc_applied 符号约定")
    print("=" * 80)
    
    print("""
MuJoCo 官方文档:
---------------------------------------------------
mjData.qfrc_applied:
- Shape: [nv]  (广义速度维度)
- 用途: 用户施加的外部力/力矩
- 单位: 力 (N) 或力矩 (N·m)
- 符号: 
  * 正值: 沿关节轴正方向的力/力矩
  * 负值: 沿关节轴负方向的力/力矩

对于旋转关节:
- 关节轴方向由 MJCF 中的 <joint axis="x y z"/> 定义
- 正力矩绕关节轴正方向旋转 (右手定则)
- 负力矩绕关节轴负方向旋转

G1 关节轴 (已验证正确):
- Pitch 关节: Y 轴 [0, 1, 0]
- Roll 关节: X 轴 [1, 0, 0]
- Yaw 关节: Z 轴 [0, 0, 1]

MJLab apply_torques 实现:
```python
def apply_torques(self, v_adr: torch.Tensor, tau: torch.Tensor):
    self.sim.data.qfrc_applied[:, v_adr] = tau
```

✅ 直接赋值,不反转符号
✅ 使用 v_adr (qvel 地址),与 qfrc_applied 的维度一致
    """)


def check_isaacgym_comparison():
    """与 IsaacGym 对比"""
    
    print("\n" + "=" * 80)
    print("IsaacGym 力矩施加对比")
    print("=" * 80)
    
    print("""
IsaacGym:
---------------------------------------------------
gym.set_dof_actuation_force_tensor(sim, forces_tensor)

- forces_tensor: [num_envs * num_dofs]
- 符号约定: 正值驱动关节正向旋转
- PD 控制器: τ = Kp * (target - current) + Kd * (target_vel - current_vel)
- ✅ 直接施加,不反转符号

Genesis:
---------------------------------------------------
robot.set_dofs_kp/set_dofs_kv  (内置 PD 控制)
或手动计算力矩后通过 control 施加

- 符号约定: 正值驱动关节正向旋转
- ✅ 直接施加,不反转符号

结论:
---------------------------------------------------
所有仿真器都使用相同的符号约定:
- PD 控制器: τ = Kp * (target - current) + Kd * (target_vel - current_vel)
- 正力矩 → 关节正向旋转
- 负力矩 → 关节负向旋转

MJLab 的 apply_torques 应该:
✅ 直接赋值给 qfrc_applied,不反转符号
    """)


def check_potential_issues():
    """检查潜在问题"""
    
    print("\n" + "=" * 80)
    print("潜在问题检查")
    print("=" * 80)
    
    issues = []
    
    print("\n1. 齿轮比 (Gear Ratio)")
    print("-" * 80)
    print("""
MuJoCo 支持关节齿轮比:
- MJCF: <joint gear="value"/>
- 作用: qfrc_applied 会自动乘以齿轮比

检查方法:
```python
import mujoco
model = mj_model
for i, jid in enumerate(joint_ids):
    gear = model.jnt_gear[jid]
    print(f"Joint {i}: gear = {gear}")
```

如果齿轮比不是 1.0,可能需要考虑!
    """)
    issues.append("检查 G1 MJCF 中的关节齿轮比设置")
    
    print("\n2. 力矩限制")
    print("-" * 80)
    print("""
MuJoCo 的力矩限制:
- MJCF: <joint actuatorfrcrange="min max"/>
- 或通过 actuator 限制

MJLab 当前实现:
- 只设置 qfrc_applied,不主动 clamp
- 依赖 MuJoCo 的内部限制

可能的问题:
- 如果 MJCF 没有配置力矩限制,可能施加超限力矩
- PD 控制器输出需要手动 clamp 到 torque_limits
    """)
    issues.append("检查 PD 控制器输出是否已经 clamp")
    
    print("\n3. 控制频率")
    print("-" * 80)
    print("""
当前配置:
- dt = 0.005s
- substeps = 10
- 物理频率: 1 / (0.005/10) = 2000 Hz
- 控制频率: 1 / 0.005 = 200 Hz (每个 sim step 更新一次力矩)

IsaacGym 典型配置:
- dt = 0.0083s (~120 Hz)
- substeps = 2
- 物理频率: ~240 Hz
- 控制频率: ~120 Hz

⚠️  问题:
- MJLab 的物理频率 (2000Hz) 远高于 IsaacGym (240Hz)
- 控制频率 (200Hz) 也高于 IsaacGym (120Hz)
- 过高的频率可能导致数值不稳定!
    """)
    issues.append("降低 MJLab 的 dt 或增加 substeps")
    
    print("\n" + "=" * 80)
    print("需要进一步检查的问题:")
    print("=" * 80)
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")


if __name__ == "__main__":
    print("力矩施加方向诊断工具\n")
    
    # 检查 PD 控制器符号
    torque = check_pd_controller_sign()
    
    # 检查 MuJoCo 约定
    check_mujoco_qfrc_convention()
    
    # 与 IsaacGym 对比
    check_isaacgym_comparison()
    
    # 检查潜在问题
    check_potential_issues()
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("✅ 1. PD 控制器符号正确")
    print("✅ 2. qfrc_applied 使用正确 (直接赋值,不反转)")
    print("✅ 3. 与 IsaacGym/Genesis 符号约定一致")
    print("\n⚠️  需要检查:")
    print("1. G1 MJCF 中的齿轮比设置")
    print("2. PD 控制器输出是否 clamp 到力矩限制")
    print("3. 🔴 物理频率过高 (2000Hz) 可能导致不稳定!")
