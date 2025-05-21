#!/usr/bin/env python3
"""Test script for bi-level safe control framework."""

import sys, os
# clear console
os.system('cls' if os.name == 'nt' else 'clear')

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
from src.simulation.simulator import Simulation
from src.simulation.scenarios import BaseScenario
from src.models.robot import Robot4WSD
from src.utils.config import Load_Config
from src.utils.visualizer import RobotVisualizer as Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main() -> int:
    """Run test simulation.
    
    Returns:
        0 on success, 1 on error
    """
    
    logger.info("Starting test simulation...")
    # Load configuration
    logger.info("Loading configuration...")
    config_path = project_root / 'config' / 'default_config.yaml'
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1
    config = Load_Config(config_path)
    # change some config values for testing
    config['controller']['type'] = "PID"

    logger.info(f"Loaded config from {config_path}")

    try:
        # Create and configure simulation
        robot = Robot4WSD(config)
        sim = Simulation(config)

        # Run simulation loop
        logger.info("Starting simulation loop ...")
        while sim.step():
            if sim.time % sim.log_dt < sim.dt:  # Log every second
                action = sim.controller.action(sim.robot.state, sim.desired_state)
                logger.info(f"T: {sim.time:.1f}, S: {sim.robot.state[:3]}, C: δ_f={action[0]:.2f}, δ_r={action[1]:.2f}, "
                             f"V=[{action[2]:.1f}, {action[3]:.1f}, {action[4]:.1f}, {action[5]:.1f}]")

            if sim.time % config['logging']['visualization_frequency'] < sim.dt:
                Visualizer.plot_results(sim)
                # plt.pause(0.001)
            
        logger.info("Simulation complete")
        
        # Save final figure
        plt.savefig('simulation_result.png', bbox_inches='tight', dpi=600)
        logger.info("Saved final figure as simulation_result.png")
        
        return 0

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return 1

if __name__ == '__main__':
    main()
