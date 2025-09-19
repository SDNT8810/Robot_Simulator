"""Basic simulation environment."""

import numpy as np
from pathlib import Path
from src.simulation.scenarios import BaseScenario
from src.controllers.planner import BasePlanner
from src.controllers.mpc import BiLevelMPC
from src.controllers.simplified_mpc import SimplifiedMPC
from src.controllers.fast_mpc import FastMPC
from src.controllers.pid import PID
from src.controllers.fuzzy import Fuzzy
from src.models.Lidar import Lidar
from src.models.robot import Robot4WSD
from src.safety.barrier import extract_safety_data

import logging

logger = logging.getLogger(__name__)

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

        # Initialize controller based on config
        ctype = config['controller']['type']
        if ctype == 'PID':
            self.controller = PID(config)
        elif ctype == 'BiLVLMPC':
            self.controller = BiLevelMPC(config)
        elif ctype == 'SimplifiedMPC':
            self.controller = SimplifiedMPC(config)
        elif ctype == 'FastMPC':
            self.controller = FastMPC(config)
        elif ctype == 'Fuzzy':
            self.controller = Fuzzy(config)
        else:
            raise ValueError(f"Unsupported controller type: {ctype}")

        # Lidar sensor setup (used by fuzzy controller; harmless otherwise)
        vis_lidar_cfg = config['visualization']['lidar_params']
        self.lidar = Lidar(num_beams=vis_lidar_cfg['num_beams'], max_range=vis_lidar_cfg['max_range'], fov=vis_lidar_cfg['fov'])
                
        # Store history for visualization
        self.input_history = {}  # Store control inputs
        self.sim_data = {
            'robot_state': self.robot.state.copy(),
            'desired_state': self.desired_state.copy(),
            'time': self.time,
            'safety_data': {}
        }
        self.safety_data = {}

        
    def step(self) -> bool:
        """Perform one simulation step."""
        self.time += self.dt
        
        self.desired_state = self.planner.get_desired_state(self.robot.state, self.scenario, self.time)
        # Extract and add safety constraint data
        # Prepare basic simulation data for logging with flat structure
        self.sim_data = {
            'time': self.time,
            # Position components (flat structure for CSV)
            'pos_x': self.robot.state[0],
            'pos_y': self.robot.state[1], 
            'pos_z': self.robot.state[2],
            # Velocity components (flat structure for CSV)
            'vel_x': self.robot.state[3],
            'vel_y': self.robot.state[4],
            'vel_z': self.robot.state[5]
        }
        self.safety_data = extract_safety_data(self)
        self.sim_data.update(self.safety_data)

        # Always collect lidar (lightweight) so fuzzy controller can use it via safety_data
        _, distances, angles = self.lidar.cast_rays(self.robot.state[0], self.robot.state[1], self.robot.state[2], obstacles=[])
        self.safety_data['lidar_distances'] = distances
        self.safety_data['lidar_angles'] = angles
        action = self.controller.action(self.robot.state, self.desired_state, self.safety_data)

        # Actuator inputs (flat structure for CSV)
        self.sim_data['actuator_front'] = action[0]
        self.sim_data['actuator_rear'] = action[1]
        # Wheel speeds (flat structure for CSV)
        self.sim_data['wheel_speed_fl'] = action[2]  # front left
        self.sim_data['wheel_speed_fr'] = action[3]  # front right
        self.sim_data['wheel_speed_rl'] = action[4]  # rear left
        self.sim_data['wheel_speed_rr'] = action[5]  # rear right

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
