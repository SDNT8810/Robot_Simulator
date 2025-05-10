"""Basic simulation scenarios."""

import numpy as np
from src.models.robot import Robot4WSD


class BaseScenario:
    """Base class for all scenarios."""
    
    def __init__(self, config: dict):
        """Initialize base scenario.
        
        Args:
            config: Configuration dictionary
        """
        # Load config and get parameters
        self.config = config
        self.duration = self.config['timing']['total_time']
        self.scenario_name = config['scenario']['name']
        
        # Initialize scenario-specific parameters
        if self.scenario_name == 'circle':
            self.radius = self.config['scenario']['circle']['radius']
            self.center = np.array(self.config['scenario']['circle']['center'])
            self.v_desired = self.config['scenario']['circle']['speed']
            self.omega_desired = self.v_desired / self.radius
            self.initial_position = np.array(self.config['scenario']['circle']['initial_position'])
            
            # Calculate required steering for circular motion
            self.wheelbase = config['robot']['wheelbase']
            self.track_width = config['robot']['track_width']
            # Pre-calculate nominal steering angles for circle
            v = self.v_desired
            if v > 0.1:  # Only when moving
                self.nominal_delta_front = np.arctan2(
                    2 * self.wheelbase * np.sin(self.omega_desired),
                    2 * self.wheelbase * np.cos(self.omega_desired) - self.track_width * self.omega_desired
                )
                self.nominal_delta_rear = np.arctan2(
                    2 * self.wheelbase * np.sin(self.omega_desired),
                    2 * self.wheelbase * np.cos(self.omega_desired) + self.track_width * self.omega_desired
                )
            else:
                self.nominal_delta_front = 0
                self.nominal_delta_rear = 0
                
        elif self.scenario_name == 'to_goal':
            self.goal = np.array(self.config['scenario']['to_goal']['goal'])
            self.initial_position = np.array(self.config['scenario']['to_goal']['initial_position'])
            self.goal_tolerance = self.config['scenario']['to_goal']['goal_tolerance']
            self.human_positions = np.array(self.config['scenario']['to_goal']['humans']['positions'])
            self.human_velocities = np.array(self.config['scenario']['to_goal']['humans']['velocities'])
            # Set a default desired velocity for to_goal scenario

    def get_desired_state(self, time: float) -> np.ndarray:
        """Get the desired state of the scenario including velocities."""
        if self.scenario_name == 'circle':
            # Calculate desired state for circular motion
            # Theta represents the angle around the circle
            theta = self.omega_desired * time
            
            # Position on circle
            x = self.center[0] + self.radius * np.cos(theta)
            y = self.center[1] + self.radius * np.sin(theta)
            
            # Heading angle (tangent to circle, plus 90 degrees)
            heading = theta + np.pi/2
            
            # Calculate desired velocities in world frame
            vx = -self.v_desired * np.sin(theta)  # x velocity component
            vy = self.v_desired * np.cos(theta)   # y velocity component
            omega = self.omega_desired           # constant angular velocity
            
            return np.array([x, y, heading, vx, vy, omega])
            
        elif self.scenario_name == 'to_goal':

            self.v_desired = self.config['scenario']['to_goal']['desired_velocity']
            dx = self.goal[0] - self.initial_position[0]
            dy = self.goal[1] - self.initial_position[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # If close to goal, stop
            if dist < self.goal_tolerance:
                return np.concatenate([self.goal, np.zeros(3)])
            
            # Calculate desired heading and velocities
            heading = np.arctan2(dy, dx)
            speed = min(self.v_desired, dist)  # Slow down near goal
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            omega = 0.0  # No rotation when going to goal
            
            return np.array([self.goal[0], self.goal[1], heading, vx, vy, omega])
        else:
            raise ValueError(f"Unknown scenario name: {self.scenario_name}")
                
    def get_initial_state(self) -> np.ndarray:
        """Get the initial state of the scenario including velocities."""
        if self.scenario_name == 'circle':
            # For circle, start with the velocity needed for circular motion
            theta_0 = np.deg2rad(self.initial_position[2])  # Convert initial angle to radians
            
            # Initial velocities tangent to circle
            vx_0 = -self.v_desired * np.sin(theta_0)
            vy_0 = self.v_desired * np.cos(theta_0)
            omega_0 = self.omega_desired
            
            return np.array([
                self.initial_position[0], 
                self.initial_position[1],
                theta_0,
                vx_0, vy_0, omega_0
            ])
            
        elif self.scenario_name == 'to_goal':
            # Start from rest for goal-directed motion
            vx_0 = self.config['scenario']['to_goal']['initial_speed'][0]
            vy_0 = self.config['scenario']['to_goal']['initial_speed'][1]
            omega_0 = self.config['scenario']['to_goal']['initial_speed'][2]
            return np.concatenate([self.initial_position, [vx_0, vy_0, omega_0]])
        else:
            raise ValueError(f"Unknown scenario name: {self.scenario_name}")

    def get_human_positions(self) -> np.ndarray:
        """Get the positions of humans in the scenario."""
        if self.scenario_name == 'to_goal':
            return self.human_positions
        else:
            return np.array([])  # No humans in other scenarios
        
