"""PID controller implementation."""

import numpy as np
from dataclasses import dataclass
from src.models.robot import Robot4WSD

@dataclass
class PIDParams:
    """PID parameters from config"""
    Kp: float  # Proportional gain
    Ki: float  # Integral gain
    Kd: float  # Derivative gain
    u_min: np.ndarray  # Input lower bounds
    u_max: np.ndarray  # Input upper bounds
    dt: float  # Time step

class PID:
    """PID control implementation."""
    
    def __init__(self, config: dict):
        """Initialize PID controller."""
        self.config = config
        
        # Set up PID parameters
        self.params = PIDParams(
            Kp=np.array(self.config['controller']['pid']['Kp']),
            Ki=np.array(self.config['controller']['pid']['Ki']),
            Kd=np.array(self.config['controller']['pid']['Kd']),
            u_min=np.array(self.config['controller']['pid']['min_output']),
            u_max=np.array(self.config['controller']['pid']['max_output']),
            dt=self.config['timing']['time_step']
        )
        
        # Initialize error terms (only for position/orientation control)
        self.state_dim = 3  # [x, y, theta] for position control
        self.int_error = np.zeros(self.state_dim)
        self.last_error = np.zeros(self.state_dim)
        
        # Get vehicle parameters
        self.wheelbase = config['robot']['wheelbase']
        self.track_width = config['robot']['track_width']
        self.max_steer = config['controller']['mpc']['constraints']['steering']['max'][0]
        
    def action(self, state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        """Compute control action using PID.
        
        Args:
            state: Robot state [x, y, theta, vx, vy, omega]
            desired_state: Desired state [x_d, y_d, theta_d, vx_d, vy_d, omega_d]
            
        Returns:
            Control action [δ_front, δ_rear, V_FL, V_FR, V_RL, V_RR]
        """
        # Extract current state
        x, y, theta = state[0:3]
        
        # Get desired position
        x_d, y_d = desired_state[0:2]
        
        # Calculate desired heading as angle to target
        dx = x_d - x
        dy = y_d - y
        theta_d = np.arctan2(dy, dx)
        
        # Calculate position error in global frame
        pos_error = np.array([dx, dy])
        
        # Transform to body frame
        R = np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])
        pos_error_body = R @ pos_error
        
        # Calculate heading error (shortest angular distance)
        theta_error = np.arctan2(np.sin(theta_d - theta), np.cos(theta_d - theta))
        
        # Combine errors
        error = np.array([pos_error_body[0], pos_error_body[1], theta_error])
        
        # Update integral and calculate derivative
        self.int_error = self.int_error + error * self.params.dt
        derivative = (error - self.last_error) / self.params.dt
        self.last_error = error.copy()
        
        # PID control for forward velocity and turning
        v_long = (self.params.Kp[0] * error[0] + 
                 self.params.Ki[0] * self.int_error[0] + 
                 self.params.Kd[0] * derivative[0])
        
        v_lat = (self.params.Kp[1] * error[1] + 
                self.params.Ki[1] * self.int_error[1] + 
                self.params.Kd[1] * derivative[1])
        
        omega = (self.params.Kp[2] * error[2] + 
                self.params.Ki[2] * self.int_error[2] + 
                self.params.Kd[2] * derivative[2])
        
        # Calculate steering angles
        if abs(v_long) > 0.1:  # Only steer when moving
            # Front steering considering both heading and lateral error
            delta_front = np.arctan2(omega * self.wheelbase + v_lat, 
                                   abs(v_long))
            
            # Rear steering for better tracking
            delta_rear = self.config['controller']['pid']['front_rear_scale'] * delta_front
        else:
            delta_front = 0
            delta_rear = 0
            
        # Clip steering angles
        delta_front = np.clip(delta_front, -self.max_steer, self.max_steer)
        delta_rear = np.clip(delta_rear, -self.max_steer, self.max_steer)
        
        # Calculate wheel velocities
        v_base = np.clip(v_long, -2.0, 2.0)
        
        if abs(delta_front) < 0.001:  # Going straight
            v_wheels = np.array([v_base] * 4)
        else:
            # Calculate wheel velocities based on turning radius
            turn_radius = abs(self.wheelbase / np.tan(delta_front))
            
            # Calculate individual wheel speeds
            wheel_radius = np.array([
                turn_radius - self.track_width/2,  # FL
                turn_radius + self.track_width/2,  # FR
                turn_radius - self.track_width/2,  # RL
                turn_radius + self.track_width/2   # RR
            ])
            
            # Apply speed ratios
            v_wheels = v_base * wheel_radius / turn_radius
            
            # Adjust for turn direction
            if delta_front < 0:  # Turning left
                v_wheels = np.array([v_wheels[1], v_wheels[0], v_wheels[3], v_wheels[2]])
        
        # Return control action
        return np.concatenate([[delta_front, delta_rear], v_wheels])
