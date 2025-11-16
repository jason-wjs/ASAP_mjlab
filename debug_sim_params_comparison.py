#!/usr/bin/env python3
"""
诊断脚本: 对比 MJLab 和 IsaacGym 的物理参数

检查点:
1. 控制频率
2. 物理频率
3. 求解器迭代次数
4. 时间步长
"""

import sys

def compare_sim_params():
    """对比仿真参数"""
    
    print("=" * 80)
    print("IsaacGym vs MJLab 物理参数对比")
    print("=" * 80)
    
    # IsaacGym 配置
    print("\nIsaacGym 配置:")
    print("-" * 80)
    ig_fps = 200
    ig_control_decimation = 4
    ig_substeps = 1
    ig_dt = 1.0 / ig_fps
    ig_control_freq = ig_fps / ig_control_decimation
    ig_physics_freq = ig_fps * ig_substeps
    ig_physics_dt = ig_dt / ig_substeps
    
    print(f"  fps: {ig_fps}")
    print(f"  control_decimation: {ig_control_decimation}")
    print(f"  substeps: {ig_substeps}")
    print(f"  dt (sim step): {ig_dt:.6f}s = {1000*ig_dt:.2f}ms")
    print(f"  physics_dt: {ig_physics_dt:.6f}s = {1000*ig_physics_dt:.2f}ms")
    print(f"  ✅ 控制频率: {ig_control_freq:.1f} Hz")
    print(f"  ✅ 物理频率: {ig_physics_freq:.1f} Hz")
    print(f"  solver iterations: 4 (position) + 0 (velocity)")
    
    # MJLab 配置 (YAML 文件)
    print("\nMJLab 配置 (YAML):")
    print("-" * 80)
    mj_dt = 0.005
    mj_substeps = 4  # YAML 中的值
    mj_control_decimation = 4  # YAML 中的值
    mj_solver_iters = 20
    
    mj_sim_freq = 1.0 / mj_dt
    mj_control_freq = mj_sim_freq / mj_control_decimation
    mj_physics_freq = mj_sim_freq * mj_substeps
    mj_physics_dt = mj_dt / mj_substeps
    
    print(f"  dt: {mj_dt:.6f}s = {1000*mj_dt:.2f}ms")
    print(f"  substeps: {mj_substeps}")
    print(f"  control_decimation: {mj_control_decimation}")
    print(f"  physics_dt: {mj_physics_dt:.6f}s = {1000*mj_physics_dt:.2f}ms")
    print(f"  ✅ 控制频率: {mj_control_freq:.1f} Hz")
    print(f"  ✅ 物理频率: {mj_physics_freq:.1f} Hz")
    print(f"  solver iterations: {mj_solver_iters}")
    
    # 对比
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    
    print(f"\n控制频率:")
    print(f"  IsaacGym: {ig_control_freq:.1f} Hz")
    print(f"  MJLab:    {mj_control_freq:.1f} Hz")
    if abs(ig_control_freq - mj_control_freq) < 1:
        print(f"  ✅ 匹配!")
    else:
        print(f"  ⚠️  差异: {abs(ig_control_freq - mj_control_freq):.1f} Hz")
    
    print(f"\n物理频率:")
    print(f"  IsaacGym: {ig_physics_freq:.1f} Hz")
    print(f"  MJLab:    {mj_physics_freq:.1f} Hz")
    if abs(ig_physics_freq - mj_physics_freq) < 10:
        print(f"  ✅ 接近!")
    else:
        print(f"  ⚠️  差异: {abs(ig_physics_freq - mj_physics_freq):.1f} Hz")
        if mj_physics_freq > ig_physics_freq:
            print(f"  🔴 MJLab 物理频率过高,可能导致数值不稳定!")
    
    print(f"\n物理时间步:")
    print(f"  IsaacGym: {ig_physics_dt:.6f}s = {1000*ig_physics_dt:.3f}ms")
    print(f"  MJLab:    {mj_physics_dt:.6f}s = {1000*mj_physics_dt:.3f}ms")
    
    print(f"\n求解器迭代:")
    print(f"  IsaacGym: 4 iterations (PGS/TGS)")
    print(f"  MJLab:    {mj_solver_iters} iterations (Newton)")
    print(f"  ℹ️  MuJoCo 使用更精确的求解器,迭代次数不直接可比")


def check_actual_runtime_params():
    """检查运行时实际参数"""
    
    print("\n" + "=" * 80)
    print("运行时参数检查")
    print("=" * 80)
    
    print("""
⚠️  重要发现:

从之前的运行日志中看到:
  njmax = 2500
  nconmax = 1000

但 YAML 配置文件中:
  njmax: 250
  nconmax: 35

这意味着:
1. 配置值被覆盖或有默认值
2. 或者某处代码硬编码了参数

需要检查:
- mjlab.py setup() 函数中的参数读取逻辑
- 是否有命令行覆盖
- MJLab backend 的默认值

另一个疑点:
- YAML: substeps: 4
- 之前诊断脚本显示: substeps = 10 (?)

需要验证运行时实际使用的 substeps 值!
    """)


def recommended_params():
    """推荐的参数配置"""
    
    print("\n" + "=" * 80)
    print("推荐的 MJLab 参数配置")
    print("=" * 80)
    
    print("""
为了匹配 IsaacGym 的稳定性,建议:

方案 1: 直接匹配 IsaacGym
----------------------------
dt: 0.005                 # 5ms sim step (200 Hz)
substeps: 1               # 无内部子步
control_decimation: 4     # 50 Hz 控制
solver_iterations: 50     # MuJoCo 默认值

结果:
- 控制频率: 50 Hz ✅
- 物理频率: 200 Hz ✅
- 与 IsaacGym 完全匹配

方案 2: 保守配置 (更稳定)
----------------------------
dt: 0.002                 # 2ms sim step (500 Hz)
substeps: 1               # 无内部子步
control_decimation: 10    # 50 Hz 控制
solver_iterations: 50

结果:
- 控制频率: 50 Hz ✅
- 物理频率: 500 Hz (比 IsaacGym 高,但 MuJoCo 可以处理)
- 更精细的物理模拟

方案 3: 当前 YAML 配置 (需验证)
----------------------------
dt: 0.005
substeps: 4
control_decimation: 4
solver_iterations: 20

结果 (如果 substeps 确实是 4):
- 控制频率: 50 Hz ✅
- 物理频率: 800 Hz (仍然偏高)
- solver_iterations 偏低

建议: 先尝试方案 1,最接近 IsaacGym
    """)
    
    print("\n修改步骤:")
    print("1. 编辑 humanoidverse/config/simulator/mjlab.yaml")
    print("2. 确认或修改参数:")
    print("   dt: 0.005")
    print("   substeps: 1")
    print("   control_decimation: 4")
    print("   solver_iterations: 50")
    print("3. 删除或确认 njmax/nconmax 值")
    print("4. 重新运行训练")


if __name__ == "__main__":
    print("MJLab vs IsaacGym 参数对比工具\n")
    
    # 对比参数
    compare_sim_params()
    
    # 检查运行时参数
    check_actual_runtime_params()
    
    # 推荐配置
    recommended_params()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("✅ 已验证正确: DOF状态、速度转换、根部参考点、力矩方向")
    print("🔴 发现问题: 物理频率配置不一致,需要对齐 IsaacGym")
    print("\n下一步: 修改 mjlab.yaml 参数并测试")
