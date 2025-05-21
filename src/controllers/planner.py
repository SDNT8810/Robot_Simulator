"""Basic Planner."""

import numpy as np
from src.models.robot import Robot4WSD
from src.simulation.scenarios import BaseScenario

class BasePlanner:
    """Base class for all planners."""
    
    def __init__(self, config: dict):
        """Initialize base planner.
        
        Args:
            config: Configuration dictionary
        """
        # Load config and get parameters
        self.config = config
        self.scenario_name = config['scenario']['name']
    
    def get_desired_state(self, state: np.ndarray, scenario: BaseScenario, time: float) -> np.ndarray:
        """Get the desired state of the scenario including velocities."""
        self.time = time
        desired_state = scenario.get_desired_state(self.time)
        if self.scenario_name == 'circle':
            # Calculate desired state for circular motion
            x = desired_state[0]
            y = desired_state[1]
            theta = desired_state[2]

            # clip theta to [-pi, pi]
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            # Heading angle (tangent to circle, plus 90 degrees)
            heading = theta + np.pi/2
            
            # Calculate desired velocities in world frame
            vx = desired_state[3]       # x velocity component
            vy = desired_state[4]       # y velocity component
            omega = desired_state[5]    # constant angular velocity
            
            return np.array([x, y, heading, vx, vy, omega])
            
        elif self.scenario_name == 'to_goal':
            self.v_desired = self.config['scenario']['to_goal']['desired_velocity']
            dx = desired_state[0] - state[0]
            dy = desired_state[1] - state[1]
            dist = np.sqrt(dx*dx + dy*dy)
            # If close to goal, stop
            if dist < scenario.goal_tolerance:
                # Use last heading or current robot heading for smooth stop
                heading = state[2] if state.shape[0] > 2 else 0.0
                # Always return a 6-element array for desired_state
                return np.array([
                    desired_state[0],
                    desired_state[1],
                    heading,
                    0.0,
                    0.0,
                    0.0
                ])
            # Calculate desired heading and velocities
            heading = np.arctan2(dy, dx)
            speed = min(self.v_desired, dist)  # Slow down near goal
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            omega = 0.0  # No rotation when going to goal
            return np.array([
                desired_state[0],
                desired_state[1],
                heading,
                vx,
                vy,
                omega
            ])
        else:
            raise ValueError(f"Unknown scenario name: {self.scenario_name}")
