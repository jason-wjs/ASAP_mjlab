#!/usr/bin/env python3
"""
诊断脚本: 验证 set_dof_state_tensor 的输入格式和 reshape 操作

检查点:
1. dof_states 的输入维度是 [num_envs, num_dof*2] 还是 [num_envs, num_dof, 2]?
2. reshape 操作是否正确提取位置和速度?
3. 与 IsaacGym 的格式对比
"""

import sys
import torch
import numpy as np

def test_dof_state_reshape():
    """测试不同输入格式的 reshape 行为"""
    
    print("=" * 80)
    print("DOF State Tensor 格式测试")
    print("=" * 80)
    
    num_envs = 2
    num_dof = 4
    
    # 格式 1: [num_envs, num_dof*2] - 交错格式 [pos0, vel0, pos1, vel1, ...]
    print("\n格式 1: [num_envs, num_dof*2] 交错格式")
    print("-" * 80)
    dof_states_interleaved = torch.tensor([
        # env 0: [pos0, vel0, pos1, vel1, pos2, vel2, pos3, vel3]
        [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        # env 1: [pos0, vel0, pos1, vel1, pos2, vel2, pos3, vel3]
        [5.0, 50.0, 6.0, 60.0, 7.0, 70.0, 8.0, 80.0],
    ])
    print(f"输入 shape: {dof_states_interleaved.shape}")
    print(f"输入内容:\n{dof_states_interleaved}")
    
    # MJLab 当前的 reshape 方式
    reshaped = dof_states_interleaved.view(num_envs, -1, 2)
    print(f"\nReshape 后 shape: {reshaped.shape}")
    print(f"Reshape 后内容:\n{reshaped}")
    print(f"提取的位置 [:, :, 0]:\n{reshaped[:, :, 0]}")
    print(f"提取的速度 [:, :, 1]:\n{reshaped[:, :, 1]}")
    
    # 格式 2: [num_envs, num_dof, 2] - 分离格式
    print("\n" + "=" * 80)
    print("格式 2: [num_envs, num_dof, 2] 分离格式")
    print("-" * 80)
    dof_states_separated = torch.tensor([
        # env 0: [[pos0, vel0], [pos1, vel1], [pos2, vel2], [pos3, vel3]]
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        # env 1: [[pos0, vel0], [pos1, vel1], [pos2, vel2], [pos3, vel3]]
        [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
    ])
    print(f"输入 shape: {dof_states_separated.shape}")
    print(f"输入内容:\n{dof_states_separated}")
    
    # MJLab 当前的 reshape 方式
    reshaped2 = dof_states_separated.view(num_envs, -1, 2)
    print(f"\nReshape 后 shape: {reshaped2.shape}")
    print(f"Reshape 后内容:\n{reshaped2}")
    print(f"提取的位置 [:, :, 0]:\n{reshaped2[:, :, 0]}")
    print(f"提取的速度 [:, :, 1]:\n{reshaped2[:, :, 1]}")
    
    # 对比两种格式
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    if torch.equal(reshaped, reshaped2):
        print("✅ 两种格式 reshape 结果相同")
    else:
        print("❌ 两种格式 reshape 结果不同!")
        print(f"格式1提取的位置:\n{reshaped[:, :, 0]}")
        print(f"格式2提取的位置:\n{reshaped2[:, :, 0]}")
    
    return dof_states_interleaved, dof_states_separated


def check_isaacgym_format():
    """检查 IsaacGym 使用的格式"""
    print("\n" + "=" * 80)
    print("IsaacGym DOF State 格式参考")
    print("=" * 80)
    
    print("""
IsaacGym 的 dof_state 格式:
- gym.get_actor_dof_states() 返回的是结构化数组
- 每个元素包含 'pos' 和 'vel' 字段
- 转换为 PyTorch tensor 时通常是 [num_envs, num_dof, 2] 格式
- 其中 [..., 0] 是位置, [..., 1] 是速度

MJLab 的 set_dof_state_tensor 函数:
```python
def set_dof_state_tensor(self, set_env_ids, dof_states):
    env_ids = set_env_ids
    dof_pos = dof_states.view(self.num_envs, -1, 2)[env_ids, :, 0]
    dof_vel = dof_states.view(self.num_envs, -1, 2)[env_ids, :, 1]
    self._backend.set_dof_state(env_ids, self._joint_q_adr, self._joint_v_adr, dof_pos, dof_vel)
```

⚠️  潜在问题:
1. 如果输入是 [num_envs, num_dof*2] 且是交错格式,view 操作会正确解析
2. 如果输入是 [num_envs, num_dof, 2],view 也会正确解析
3. 但如果输入已经是 [num_envs, num_dof, 2],再 view 可能不必要但无害

需要检查调用处传入的 dof_states 格式!
    """)


def trace_dof_state_usage():
    """追踪 dof_states 在环境中的使用"""
    print("\n" + "=" * 80)
    print("需要检查的代码位置")
    print("=" * 80)
    
    locations = [
        "1. 环境 reset 时调用 set_dof_state_tensor",
        "2. 检查 self.dof_state 属性的构造",
        "3. 检查任何直接修改 dof_states 的地方",
        "4. 对比 IsaacGym/Genesis 的实现",
    ]
    
    for loc in locations:
        print(f"  {loc}")
    
    print("\n关键搜索命令:")
    print("  grep -r 'set_dof_state_tensor' humanoidverse/")
    print("  grep -r 'dof_state' humanoidverse/envs/")
    print("  grep -r 'self.dof_state' humanoidverse/")


if __name__ == "__main__":
    print("DOF State 格式诊断工具\n")
    
    # 测试 reshape 行为
    test_dof_state_reshape()
    
    # 显示 IsaacGym 参考
    check_isaacgym_format()
    
    # 显示需要检查的位置
    trace_dof_state_usage()
    
    print("\n" + "=" * 80)
    print("下一步操作:")
    print("=" * 80)
    print("1. 搜索代码中 set_dof_state_tensor 的调用位置")
    print("2. 检查传入的 dof_states 的实际维度")
    print("3. 添加断言或日志验证格式")
    print("4. 对比 IsaacGym 实现确认预期格式")
