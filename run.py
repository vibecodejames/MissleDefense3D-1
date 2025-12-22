#!/usr/bin/env python3
"""
Missile Defense Simulation - Entry Point

Run this file to launch the saturation attack demonstration:
    python run.py           # 2D animation
    python run.py --3d      # 3D animated visualization with rotating camera

For the original single-target demo:
    python run.py --single
"""

import sys

# Import from the main simulation file
from Misslesim1draft import run_saturation_demo, main


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        print("Running single-target intercept demo...")
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == "--3d":
        print("Running 3D saturation attack demo...")
        run_saturation_demo(show_3d=True)
    else:
        print("Running saturation attack demo (2D animation)...")
        run_saturation_demo(show_3d=False)
