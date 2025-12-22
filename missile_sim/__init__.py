"""
Missile Simulation Package

A comprehensive missile defense simulation with:
- Realistic flight dynamics (gravity, drag, g-limits)
- Multi-phase guidance systems (boost, midcourse, terminal)
- Proportional navigation with seeker modeling
- Saturation attack scenarios

Usage:
    from missile_sim import SaturationSimulation, animate_saturation_attack

    sim = SaturationSimulation(num_targets=6, num_interceptors=8)
    sim.run()
    animate_saturation_attack(sim)
"""

# Import core components
from .core import (
    # Enums
    GuidancePhase,
    SeekerState,
    InterceptResult,

    # Data structures
    Aircraft,
    Seeker,
    TargetEstimator,
    GuidanceController,
    TargetTrack,
    InterceptorStatus,
    SAMBattery,

    # Math functions
    calculate_intercept_time,
    calculate_lead_angle,
    analyze_engagement,
)

# Version
__version__ = "1.0.0"
__author__ = "James"

# Define what gets exported with "from missile_sim import *"
__all__ = [
    # Enums
    'GuidancePhase',
    'SeekerState',
    'InterceptResult',

    # Data structures
    'Aircraft',
    'Seeker',
    'TargetEstimator',
    'GuidanceController',
    'TargetTrack',
    'InterceptorStatus',
    'SAMBattery',

    # Math functions
    'calculate_intercept_time',
    'calculate_lead_angle',
    'analyze_engagement',
]
