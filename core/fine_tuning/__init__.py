"""
Fine-tuning core module.
Group 68: Mixture-of-Experts (MoE) implementation.
"""

from .moe import MoESystem, MoEExpert, ExpertRouter, create_moe_system

__all__ = [
    'MoESystem',
    'MoEExpert',
    'ExpertRouter',
    'create_moe_system'
]
