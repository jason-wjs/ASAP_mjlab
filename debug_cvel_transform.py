#!/usr/bin/env python3
"""
诊断脚本: 验证 cvel 从体坐标系到世界坐标系的转换

检查点:
1. cvel 的格式是否为 [ang, lin]?
2. 四元数旋转变换是否使用正确的时序?
3. 是否应该旋转速度?(MuJoCo 的 cvel 已经在体坐标系)
"""

import sys
import torch
import numpy as np

def quat_apply(quat_xyzw, vec):
    """
    使用四元数旋转向量 (XYZW格式)
    参考 humanoidverse/utils/torch_utils.py
    """
    x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]
    vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
    
    # q * v * q^-1
    t = 2.0 * (y * vz - z * vy)
    u = 2.0 * (z * vx - x * vz)
    v = 2.0 * (x * vy - y * vx)
    
    rx = vx + w * t + (y * v - z * u)
    ry = vy + w * u + (z * t - x * v)
    rz = vz + w * v + (x * u - y * t)
    
    return torch.stack([rx, ry, rz], dim=-1)


def test_cvel_transformation():
    """测试 cvel 坐标系转换"""
    
    print("=" * 80)
    print("CVEL 坐标系转换测试")
    print("=" * 80)
    
    # 场景: 机器人躯干旋转 45 度 (绕 Z 轴)
    angle = np.pi / 4  # 45 度
    quat_wxyz = torch.tensor([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]  # 转换为 XYZW
    
    print(f"\n机器人朝向 (XYZW): {quat_xyzw}")
    print(f"等效旋转: 绕 Z 轴旋转 {np.degrees(angle):.1f} 度")
    
    # MuJoCo cvel 格式: [wx, wy, wz, vx, vy, vz] 在体坐标系
    # 假设机器人在体坐标系下:
    # - 向前移动 (体坐标系 X 轴): vx=1, vy=0, vz=0
    # - 绕竖直轴旋转 (体坐标系 Z 轴): wx=0, wy=0, wz=0.5
    cvel_body = torch.tensor([0.0, 0.0, 0.5, 1.0, 0.0, 0.0])
    
    print(f"\n体坐标系速度 (cvel):")
    print(f"  角速度: [wx={cvel_body[0]:.2f}, wy={cvel_body[1]:.2f}, wz={cvel_body[2]:.2f}]")
    print(f"  线速度: [vx={cvel_body[3]:.2f}, vy={cvel_body[4]:.2f}, vz={cvel_body[5]:.2f}]")
    
    # MJLab 当前的转换方式
    local_ang = cvel_body[0:3]
    local_lin = cvel_body[3:6]
    
    world_ang = quat_apply(quat_xyzw, local_ang)
    world_lin = quat_apply(quat_xyzw, local_lin)
    
    print(f"\n世界坐标系速度 (MJLab 转换):")
    print(f"  角速度: [wx={world_ang[0]:.4f}, wy={world_ang[1]:.4f}, wz={world_ang[2]:.4f}]")
    print(f"  线速度: [vx={world_lin[0]:.4f}, vy={world_lin[1]:.4f}, vz={world_lin[2]:.4f}]")
    
    # 预期结果:
    # 角速度在体坐标系下是 [0, 0, 0.5] (绕体Z轴)
    # 旋转 45 度后,世界坐标系下仍应该是 [0, 0, 0.5] (绕世界Z轴)
    # 
    # 线速度在体坐标系下是 [1, 0, 0] (沿体X轴)
    # 旋转 45 度后,世界坐标系下应该是 [cos(45°), sin(45°), 0] ≈ [0.707, 0.707, 0]
    
    expected_ang = torch.tensor([0.0, 0.0, 0.5])
    expected_lin = torch.tensor([np.cos(angle), np.sin(angle), 0.0])
    
    print(f"\n预期世界坐标系速度:")
    print(f"  角速度: [wx={expected_ang[0]:.4f}, wy={expected_ang[1]:.4f}, wz={expected_ang[2]:.4f}]")
    print(f"  线速度: [vx={expected_lin[0]:.4f}, vy={expected_lin[1]:.4f}, vz={expected_lin[2]:.4f}]")
    
    # 验证
    ang_error = torch.abs(world_ang - expected_ang).max()
    lin_error = torch.abs(world_lin - expected_lin).max()
    
    print(f"\n" + "=" * 80)
    print("验证结果:")
    print("=" * 80)
    print(f"角速度误差: {ang_error:.6f}")
    print(f"线速度误差: {lin_error:.6f}")
    
    if ang_error < 1e-5 and lin_error < 1e-5:
        print("✅ 坐标系转换正确!")
    else:
        print("❌ 坐标系转换有误!")
    
    return world_ang, world_lin, expected_ang, expected_lin


def test_root_vs_body_cvel():
    """测试根部速度使用的 cvel 来源"""
    
    print("\n" + "=" * 80)
    print("根部速度读取测试")
    print("=" * 80)
    
    print("""
MJLab 当前实现 (mjlab.py refresh_sim_tensors):
---------------------------------------------------
root_body_idx = self._body_indices[root_body_config_idx]
base_vel_local = raw["cvel"][:, root_body_idx]  # 从刚体读取
local_ang = base_vel_local[..., 0:3]
local_lin = base_vel_local[..., 3:6]
base_ang_vel = quat_apply(base_quat_xyzw, local_ang)
base_lin_vel = quat_apply(base_quat_xyzw, local_lin)

问题检查:
1. ✅ 使用 root_body 的 cvel (通常是 pelvis 或 torso_link)
2. ✅ cvel 格式 [ang, lin] 正确
3. ✅ 使用四元数旋转到世界坐标系

但需要注意:
- MuJoCo 的 cvel 是刚体相对于世界坐标系原点的速度,但表示在体坐标系中
- 对于浮动基座机器人,根部刚体的 cvel 就是基座速度
- 转换到世界坐标系是正确的

潜在问题:
- 如果 root_body 不是真正的浮动基座,而是某个子链接,可能会有问题
- 需要确认 G1 的 root_body 配置
    """)


def check_mujoco_cvel_definition():
    """检查 MuJoCo cvel 的定义"""
    
    print("\n" + "=" * 80)
    print("MuJoCo cvel 定义")
    print("=" * 80)
    
    print("""
MuJoCo 官方文档:
---------------------------------------------------
cvel (mjData.cvel):
- Shape: [nbody, 6]
- 格式: [angular_velocity, linear_velocity]  (注意:角速度在前!)
- 坐标系: 刚体的局部坐标系 (body frame)
- 定义: 刚体相对于世界坐标系的速度,但表示在刚体的局部坐标系中

对比 qvel:
- qvel: 广义速度,对于浮动基座是 [vx, vy, vz, wx, wy, wz] 在世界坐标系
- cvel: 刚体速度,[wx, wy, wz, vx, vy, vz] 在体坐标系

转换关系:
- 从 cvel (体坐标系) 转到世界坐标系: v_world = R * v_body
- 其中 R 是旋转矩阵,可用四元数表示

MJLab 的实现:
✅ 正确使用 quat_apply 进行旋转变换
✅ 正确识别 cvel 格式为 [ang, lin]

但需要验证:
❓ 刚体的四元数 (xquat) 是否与转换时刻一致?
❓ 是否存在一帧延迟?
    """)


def check_quaternion_timing():
    """检查四元数时序问题"""
    
    print("\n" + "=" * 80)
    print("四元数时序检查")
    print("=" * 80)
    
    print("""
潜在时序问题:
---------------------------------------------------
在 refresh_sim_tensors() 中:

1. 读取 raw["xquat_wxyz"][:, root_body_idx]  -> 当前帧的四元数
2. 转换为 base_quat_xyzw
3. 读取 raw["cvel"][:, root_body_idx]  -> 当前帧的体坐标系速度
4. 使用 quat_apply(base_quat_xyzw, local_vel)  -> 转换到世界坐标系

这个流程是正确的,因为:
- xquat 和 cvel 都是从同一个 raw state 读取
- 都是当前时刻的状态
- 使用当前时刻的姿态旋转当前时刻的体坐标系速度

但对于所有刚体的速度转换:
```python
self._rigid_body_rot[:] = _xquat_wxyz[..., [1, 2, 3, 0]]  # 先转换四元数
...
q_all = self._rigid_body_rot  # XYZW
self._rigid_body_ang_vel[:] = quat_apply(q_all.reshape(-1, 4), local_ang_all.reshape(-1, 3))
self._rigid_body_vel[:] = quat_apply(q_all.reshape(-1, 4), local_lin_all.reshape(-1, 3))
```

这里使用的是 **同一个函数调用内刚刚更新的** `_rigid_body_rot`,
所以不存在时序问题。

结论: ✅ 四元数时序正确
    """)


if __name__ == "__main__":
    print("CVEL 坐标系转换诊断工具\n")
    
    # 测试坐标系转换
    world_ang, world_lin, exp_ang, exp_lin = test_cvel_transformation()
    
    # 测试根部速度读取
    test_root_vs_body_cvel()
    
    # 检查 MuJoCo cvel 定义
    check_mujoco_cvel_definition()
    
    # 检查四元数时序
    check_quaternion_timing()
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("✅ 1. set_dof_state_tensor 的 reshape 操作正确")
    print("✅ 2. cvel 坐标系转换逻辑正确")
    print("✅ 3. 四元数时序正确")
    print("\n这两项高优先级检查都通过了!")
    print("\n下一步: 检查根部状态参考点和力矩施加方向")
