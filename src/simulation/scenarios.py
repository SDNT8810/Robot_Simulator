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
            # if v > 0.1:  # Only when moving
            self.nominal_delta_front = np.arctan2(
                2 * self.wheelbase * np.sin(self.omega_desired),
                2 * self.wheelbase * np.cos(self.omega_desired) - self.track_width * self.omega_desired
            )
            self.nominal_delta_rear = np.arctan2(
                2 * self.wheelbase * np.sin(self.omega_desired),
                2 * self.wheelbase * np.cos(self.omega_desired) + self.track_width * self.omega_desired
            )
            # else:
            #     self.nominal_delta_front = 0
            #     self.nominal_delta_rear = 0
                
        elif self.scenario_name == 'to_goal':
            self.goal = np.array(self.config['scenario']['to_goal']['goal'])
            self.initial_position = np.array(self.config['scenario']['to_goal']['initial_position'])
            self.goal_tolerance = self.config['scenario']['to_goal']['goal_tolerance']
        # Extract human states from scenario
        self.human_states = []
        scenario_config = self.config.get('scenario', {}).get(self.scenario_name, {})
        if 'humans' in scenario_config:
            human_positions = scenario_config['humans'].get('positions', [])
            human_velocities = scenario_config['humans'].get('velocities', [])
            for pos, vel in zip(human_positions, human_velocities):
                self.human_states.append({
                    'position': np.array(pos),
                    'velocity': np.array(vel)
                })
                    
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
            
            # Calculate desired heading (always point towards goal)
            desired_heading = np.arctan2(dy, dx)
            
            # If very close to goal, smoothly slow down and stop with precise orientation
            if dist < self.goal_tolerance * 2:  # Start slowing down earlier (2x tolerance)
                # Calculate desired deceleration profile
                max_decel = 0.5  # m/s²
                speed = min(self.v_desired, np.sqrt(2 * max_decel * dist))
                
                # Final approach - align with goal orientation if specified
                if len(self.goal) > 2:  # If goal includes orientation
                    final_heading = self.goal[2]
                else:
                    final_heading = desired_heading
                
                # Smooth transition to final orientation
                if dist < self.goal_tolerance:
                    # At goal, ensure zero velocity and exact position
                    return np.array([
                        self.goal[0],
                        self.goal[1],
                        final_heading,
                        0.0,  # Zero velocity
                        0.0,
                        0.0
                    ])
                else:
                    # Approaching goal - reduce speed and align heading
                    return np.array([
                        self.goal[0],
                        self.goal[1],
                        final_heading,  # Start aligning to final heading
                        speed * np.cos(desired_heading),  # Reduced velocity
                        speed * np.sin(desired_heading),
                        0.0  # No rotation
                    ])
            
            # Calculate desired speed with smooth deceleration
            # Start slowing down when within 2x the goal tolerance
            decel_distance = 2.0 * self.goal_tolerance
            if dist < decel_distance:
                # Smooth deceleration profile (quadratic)
                speed = self.v_desired * (dist / decel_distance) ** 2
            else:
                speed = self.v_desired
                
            # Calculate velocity components
            vx = speed * np.cos(desired_heading)
            vy = speed * np.sin(desired_heading)
            
            return np.array([
                self.goal[0],  # Always command to the goal position
                self.goal[1],
                desired_heading,  # Face the goal
                vx,
                vy,
                0.0  # No rotation
            ])
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
        # Check if current scenario has humans configuration
        scenario_config = self.config['scenario'].get(self.scenario_name, {})
        if 'humans' in scenario_config and 'positions' in scenario_config['humans']:
            return np.array(scenario_config['humans']['positions'])
        else:
            return np.array([])  # No humans configured for this scenario

