"""Basic simulation environment."""

import numpy as np
from pathlib import Path
from src.simulation.scenarios import BaseScenario
from src.controllers.planner import BasePlanner
from src.controllers.mpc import BiLevelMPC
from src.controllers.simplified_mpc import SimplifiedMPC
from src.controllers.fast_mpc import FastMPC
from src.controllers.pid import PID
from src.models.robot import Robot4WSD

class Simulation:
    """Basic simulation class."""
    
    def __init__(self, config: dict):
        """Initialize simulation."""
        self.scenario = BaseScenario(config)
        self.time = 0.0
        self.log_dt = config['logging']['log_frequency']
        self.save_run_dt = config['logging']['saving_run_frequency']
        self.planner = BasePlanner(config)
        self.dt = config['timing']['time_step']
        self.robot = Robot4WSD(config)
        self.robot.state = self.scenario.get_initial_state()
        self.desired_state = self.scenario.get_desired_state(self.time)
        self.duration = self.scenario.duration
        self.config = config
        self.action = np.concatenate([[0, 0], [0, 0, 0, 0]])
        self.safety_barrier = None

        # Initialize controller based on config
        if config['controller']['type'] == 'PID':
            self.controller = PID(config)
        elif config['controller']['type'] == 'BiLVLMPC':
            self.controller = BiLevelMPC(config)
        elif config['controller']['type'] == 'SimplifiedMPC':
            self.controller = SimplifiedMPC(config)
        elif config['controller']['type'] == 'FastMPC':
            self.controller = FastMPC(config)
        else:
            raise ValueError(f"Unsupported controller type: {config['controller']['type']}")
                
        # Store history for visualization
        self.input_history = {}  # Store control inputs
        
    def step(self) -> bool:
        """Perform one simulation step."""
        self.time += self.dt
        
        self.desired_state = self.planner.get_desired_state(self.robot.state, self.scenario, self.time)
        action = self.controller.action(self.robot.state, self.desired_state)
        
        # Store control inputs for visualization
        self.input_history[self.time] = action.copy()

        # Check if simulation is still running
        if self.is_running():
            self.robot.update(action)
            self.robot.state_history[self.time] = self.robot.state.copy()
            return True
        else:
            return False

    def is_running(self) -> bool:
        """Check if the scenario is still running."""
        if self.scenario.scenario_name == 'circle':
            return self.time < self.duration
        elif self.scenario.scenario_name == 'to_goal':
            # Check if the robot has reached the goal
            distance_to_goal = np.linalg.norm(self.robot.state[:2] - self.scenario.goal[:2])
            # Stop if we reach the goal, continue if we haven't AND we're within time limit
            return distance_to_goal > self.scenario.goal_tolerance and self.time < self.duration
        else:
            raise ValueError(f"Unknown scenario name: {self.scenario.scenario_name}")
