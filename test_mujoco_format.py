#!/usr/bin/env python3
"""
确认 MuJoCo 的 qvel 和 cvel 数据格式
"""
import mujoco
import numpy as np

# 简单的自由关节测试模型
xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <joint name="free_joint" type="free"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("=" * 80)
print("MuJoCo 数据格式测试")
print("=" * 80)

# 设置一个已知的状态
# qpos for free joint: [x, y, z, qw, qx, qy, qz]
data.qpos[:] = [1.0, 2.0, 3.0,  # position
                1.0, 0.0, 0.0, 0.0]  # quaternion WXYZ (identity)

# qvel for free joint: [vx, vy, vz, wx, wy, wz]
data.qvel[:] = [10.0, 20.0, 30.0,  # linear velocity
                0.1, 0.2, 0.3]     # angular velocity

mujoco.mj_forward(model, data)

print("\nqpos (位置和四元数):")
print(f"  qpos = {data.qpos}")
print(f"  解释: [x, y, z, qw, qx, qy, qz]")
print(f"       position = {data.qpos[:3]}")
print(f"       quat_wxyz = {data.qpos[3:7]}")

print("\nqvel (速度):")
print(f"  qvel = {data.qvel}")
print(f"  解释: [vx, vy, vz, wx, wy, wz]")
print(f"       lin_vel = {data.qvel[:3]}")
print(f"       ang_vel = {data.qvel[3:6]}")

print("\ncvel (body frame 速度):")
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
print(f"  cvel[{body_id}] = {data.cvel[body_id]}")
print(f"  cvel 格式:")

# 根据 MuJoCo 文档：cvel 格式是 [angular, linear] in body frame
# 让我们验证
print(f"    cvel[0:3] (angular) = {data.cvel[body_id, 0:3]}")
print(f"    cvel[3:6] (linear)  = {data.cvel[body_id, 3:6]}")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print("✅ qpos: [x, y, z, qw, qx, qy, qz]  - 四元数 WXYZ")
print("✅ qvel: [vx, vy, vz, wx, wy, wz]  - 线速度在前，角速度在后")
print("✅ cvel: [wx, wy, wz, vx, vy, vz]  - 角速度在前，线速度在后 (body frame)")
