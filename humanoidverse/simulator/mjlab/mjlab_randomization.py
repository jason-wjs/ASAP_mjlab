from typing import Tuple

import torch


def init_dr_tensors(num_envs: int, num_bodies: int, link_names_len: int, device: str):
    """
    Initialize domain-randomization observable tensors with no-op defaults to satisfy env expectations.
    Returns: (_base_com_bias, _link_mass_scale, friction_coeffs, restitution_coeffs, _ground_friction_values)
    """
    base_com_bias = torch.zeros(num_envs, 3, dtype=torch.float, device=device)
    link_mass_scale = torch.ones(num_envs, link_names_len, dtype=torch.float, device=device) if link_names_len > 0 else torch.ones(num_envs, 0, dtype=torch.float, device=device)
    friction_coeffs = torch.ones(num_envs, 1, 1, dtype=torch.float, device=device)
    restitution_coeffs = torch.ones(num_envs, 1, 1, dtype=torch.float, device=device)
    ground_friction_values = torch.zeros(num_envs, num_bodies, dtype=torch.float, device=device)
    return base_com_bias, link_mass_scale, friction_coeffs, restitution_coeffs, ground_friction_values

