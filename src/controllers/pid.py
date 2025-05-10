"""PID controller implementation."""

import numpy as np
from dataclasses import dataclass
from src.models.robot import Robot4WSD

@dataclass
class PIDParams:
    Kp: float  # Proportional gain
    Ki: float  # Integral gain
    Kd: float  # Derivative gain
    u_min: np.ndarray  # Input lower bounds
    u_max: np.ndarray  # Input upper bounds
    dt: float  # Time step

class PID:
    def __init__(self, config: dict):
        self.config = config
        self.params = PIDParams(
            Kp=np.array(self.config['controller']['pid']['Kp']),
            Ki=np.array(self.config['controller']['pid']['Ki']),
            Kd=np.array(self.config['controller']['pid']['Kd']),
            u_min=np.array(self.config['controller']['pid']['min_output']),
            u_max=np.array(self.config['controller']['pid']['max_output']),
            dt=self.config['timing']['time_step']
        )
        
        # Initialize state variables
        self.state_dim = 3  # [x, y, theta]
        self.int_error = np.zeros(self.state_dim)
        self.last_error = np.zeros(self.state_dim)
        self.wheelbase = config['robot']['wheelbase']
        self.track_width = config['robot']['track_width']
        self.max_steer = config['controller']['mpc']['constraints']['steering']['max'][0]
        self.prev_delta_front = 0.0
        self.prev_delta_rear = 0.0
        self.prev_voltages = np.zeros(4)
        self.prev_omega = 0.0

    def action(self, state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        # Extract state and calculate errors
        x, y, theta = state[0:3]
        x_d, y_d = desired_state[0:2]
        dx, dy = x_d - x, y_d - y
        distance = np.linalg.norm([dx, dy])
        theta_d = np.arctan2(dy, dx)
        
        # Transform to body frame
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        pos_error_body = R @ np.array([dx, dy])
        theta_error = np.arctan2(np.sin(theta_d - theta), np.cos(theta_d - theta))
        error = np.array([pos_error_body[0], pos_error_body[1], theta_error])
        
        # PID calculation
        derivative = (error - self.last_error) / self.params.dt
        self.int_error += error * self.params.dt
        self.last_error = error.copy()
        
        # Anti-windup for integral term
        windup_limit = self.config['controller']['pid'].get('int_windup_limit', 10.0)
        self.int_error = np.clip(self.int_error, -windup_limit, windup_limit)
        
        # Compute PID outputs
        v_long = self.params.Kp[0] * error[0] + self.params.Ki[0] * self.int_error[0] + self.params.Kd[0] * derivative[0]
        v_lat  = self.params.Kp[1] * error[1] + self.params.Ki[1] * self.int_error[1] + self.params.Kd[1] * derivative[1]
        omega  = self.params.Kp[2] * error[2] + self.params.Ki[2] * self.int_error[2] + self.params.Kd[2] * derivative[2]
                
        # Calculate steering angles
        delta_front = np.arctan2(omega * self.wheelbase + v_lat, abs(v_long))
        delta_rear = - 0.85 * delta_front

        # Clip steering angles
        delta_front = np.clip(delta_front, -self.max_steer, self.max_steer)
        delta_rear = np.clip(delta_rear, -self.max_steer, self.max_steer)
        
        # Calculate wheel velocities
        V_min = self.config['safety']['limits']['velocity']['min']
        V_max = self.config['safety']['limits']['velocity']['max']
        v_base = np.clip(v_long, V_min, V_max)
        
        # Calculate individual wheel velocities for turning
        if abs(delta_front) < 0.001:
            v_wheels = np.array([v_base] * 4)
        else:
            try:
                turn_radius = abs(self.wheelbase / np.tan(delta_front)) if abs(delta_front) > 0.001 else float('inf')
                wheel_radius = np.array([
                    max(0.1, turn_radius - self.track_width/2),  # FL
                    max(0.1, turn_radius + self.track_width/2),  # FR
                    max(0.1, turn_radius - self.track_width/2),  # RL
                    max(0.1, turn_radius + self.track_width/2)   # RR
                ])
                v_wheels = v_base * wheel_radius / turn_radius
            except:
                v_wheels = np.array([v_base] * 4)
        
        # Convert to voltages
        motor_voltages = v_wheels * self.config['robot']['motor']['voltage_speed_factor']

        v_motor_min = self.config['controller']['pid']['min_output']
        v_motor_max = self.config['controller']['pid']['max_output']
        motor_voltages = np.clip(motor_voltages, v_motor_min, v_motor_max)
            
        self.prev_voltages = motor_voltages.copy()
            
        # Return control action [δ_front, δ_rear, V_FL, V_FR, V_RL, V_RR]
        return np.concatenate([[delta_front, delta_rear], motor_voltages])
