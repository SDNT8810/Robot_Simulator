"""Simple PID controller implementation for robot control.

This module implements a basic PID controller that takes global position error 
as input and produces steering angles and motor voltages as output.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from src.safety.barrier import DistanceBarrier, YieldingBarrier, SpeedBarrier, AccelBarrier

@dataclass
class PIDParams:
    """Parameters for the PID controller."""
    Kp: np.ndarray  # Proportional gain for [x, y, theta]
    Ki: np.ndarray  # Integral gain for [x, y, theta]
    Kd: np.ndarray  # Derivative gain for [x, y, theta]
    u_min: float    # Minimum output (voltage)
    u_max: float    # Maximum output (voltage)
    dt: float       # Time step
    deadband: np.ndarray  # Deadband for [x, y, theta] to prevent oscillations

class PID:
    """Basic PID controller for robot steering and motor control.
    
    The controller converts position and orientation errors into
    steering angles and motor voltages with minimal filtering.
    """
    
    def __init__(self, config: dict):
        """Initialize the PID controller with configuration parameters.
        
        Args:
            config: Configuration dictionary containing controller parameters
        """
        self.config = config
        
        # Load PID parameters from config
        self.params = PIDParams(
            Kp=np.array(self.config['controller']['pid']['Kp']),
            Ki=np.array(self.config['controller']['pid']['Ki']),
            Kd=np.array(self.config['controller']['pid']['Kd']),
            u_min=self.config['controller']['pid']['min_output'],
            u_max=self.config['controller']['pid']['max_output'],
            dt=self.config['timing']['time_step'],
            deadband=np.array(self.config['controller']['pid'].get('deadband', [0.05, 0.05, 0.01]))
        )
        
        # Load robot physical parameters
        self.wheelbase = config['robot']['wheelbase']
        self.track_width = config['robot']['track_width']
        self.max_steer = config['controller']['mpc']['constraints']['steering']['max'][0]
        
        # Initialize controller state
        self.state_dim = 3  # [x, y, theta]
        self.int_error = np.zeros(self.state_dim)
        self.last_error = np.zeros(self.state_dim)
        self.prev_voltages = np.zeros(4)  # Previous voltage values
        self.prev_steering = np.zeros(2)  # Previous steering angles [front, rear]
        self.steering_deadband = 0.03  # Steering deadband in radians (about 1.7 degrees)
        # Initialize concrete barrier functions
        self.safety_barriers = [
            DistanceBarrier(config),
            YieldingBarrier(config),
            SpeedBarrier(config),
            AccelBarrier(config)
        ]
    def action(self, state: np.ndarray, desired_state: np.ndarray, safety_data: Dict[str, Any]) -> np.ndarray:
        """Compute control action using PID.
        
        Args:
            state: Current robot state [x, y, θ, vx, vy, omega]
            desired_state: Desired robot state [x, y, θ, vx, vy, omega]
            safety_data: Safety-related data for the current simulation step

                  
        Returns:
            Control action [δ_front, δ_rear, V_FL, V_FR, V_RL, V_RR]
            where δ are steering angles in radians and V are motor voltages.
        """
        # Extract current states
        x, y, theta = state[0:3]
        vx, vy, omega = state[3:6]
        
        # Extract desired states
        x_d, y_d = desired_state[0:2]
        
        # Calculate errors in global frame
        dx, dy = x_d - x, y_d - y
        
        # Convert errors to body frame for better control
        rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        pos_error_body = rotation_matrix @ np.array([dx, dy])
        
        # Calculate heading error
        theta_d = np.arctan2(dy, dx)
        theta_error = np.arctan2(np.sin(theta_d - theta), np.cos(theta_d - theta))
        
        # Combine errors
        error = np.array([pos_error_body[0], pos_error_body[1], theta_error])
        
        # Apply deadband to prevent oscillations when very close to target
        error_magnitude = np.linalg.norm(error[:2])  # Position error magnitude
        
        # Enhanced deadband handling - gradually reduce error impact near deadband
        for i in range(self.state_dim):
            if abs(error[i]) < self.params.deadband[i]:
                # Zero out errors smaller than the deadband
                error[i] = 0.0
            elif abs(error[i]) < self.params.deadband[i] * 2:
                # Gradually scale up errors near the deadband to prevent sudden jumps
                scale_factor = (abs(error[i]) - self.params.deadband[i]) / self.params.deadband[i]
                error[i] *= scale_factor
        
        # Calculate error-proportional derivative gain scaling
        # Gradually reduce derivative gain as error gets smaller to prevent oscillations
        error_scale = np.ones(self.state_dim)
        for i in range(self.state_dim):
            # Scale derivative gain down when error is small (but not zero)
            if 0 < abs(error[i]) < self.params.deadband[i] * 5:
                error_scale[i] = abs(error[i]) / (self.params.deadband[i] * 5)
        
        # Calculate derivative term with scaled gains when close to target
        derivative = (error - self.last_error) / self.params.dt
        
        # Update integral term with basic anti-windup
        for i in range(self.state_dim):
            self.int_error[i] += error[i] * self.params.dt
        
        # Apply simple anti-windup
        windup_limit = self.config['controller']['pid'].get('int_windup_limit', 3.0)
        self.int_error = np.clip(self.int_error, -windup_limit, windup_limit)
        
        # Store current error for next iteration
        self.last_error = error.copy()
        
        # Calculate PID outputs with scaling for smoother response
        v_long = self.params.Kp[0] * error[0] + self.params.Ki[0] * self.int_error[0] + self.params.Kd[0] * derivative[0] * error_scale[0]
        v_lat = self.params.Kp[1] * error[1] + self.params.Ki[1] * self.int_error[1] + self.params.Kd[1] * derivative[1] * error_scale[1]
        omega = self.params.Kp[2] * error[2] + self.params.Ki[2] * self.int_error[2] + self.params.Kd[2] * derivative[2] * error_scale[2]
        
        # Calculate vehicle speed
        vehicle_speed = np.sqrt(vx**2 + vy**2)

        # Apply nonlinear smooth response curve that reduces output as we get closer to target
        error_magnitude = np.sqrt(error[0]**2 + error[1]**2)
        
        # Enhanced target proximity handling
        # As we get closer to the target, reduce controller aggressiveness
        target_proximity_factor = min(1.0, error_magnitude / 0.5)  # Scale from 0-1 based on closeness to target
        
        # Further reduce omega response when close to target to prevent oscillation
        if error_magnitude < 0.3:
            omega *= target_proximity_factor
        
        # Apply feed-forward from desired state with proximity-based scaling
        # When close to target, feed-forward effect is reduced to prevent overshooting
        feed_forward = self.config['controller']['pid'].get('feed_forward_factor', 0.5) * target_proximity_factor
        desired_speed = np.sqrt(desired_state[3]**2 + desired_state[4]**2)
        v_long += feed_forward * desired_speed
        omega += feed_forward * desired_state[5]  # Use desired omega directly
        
        # Calculate steering angles with smoothing
        delta_front, delta_rear = self._calculate_steering(v_long, v_lat, omega, vehicle_speed)
        
        # Calculate wheel speeds and motor voltages
        motor_voltages = self._calculate_motor_voltages(v_long, delta_front, vehicle_speed)
        
        # Return control action [δ_front, δ_rear, V_FL, V_FR, V_RL, V_RR]
        return np.concatenate([[delta_front, delta_rear], motor_voltages])
    
    def _calculate_steering(self, v_long, v_lat, omega, vehicle_speed):
        """Calculate steering angles using bicycle model.
        
        Args:
            v_long: Longitudinal velocity
            v_lat: Lateral velocity
            omega: Angular velocity
            vehicle_speed: Current vehicle speed
            
        Returns:
            Tuple of (delta_front, delta_rear)
        """
        # Use bicycle model for steering angle calculation
        v_long_sign = np.sign(v_long) if abs(v_long) > 0.1 else 1.0
        
        # Calculate front steering angle        
        delta_front = np.arctan2(omega * self.wheelbase + v_lat, v_long_sign * abs(v_long))
        
        # Clip steering to physical limits
        delta_front = np.clip(delta_front, -self.max_steer, self.max_steer)
        
        # Simple front/rear steering ratio
        front_rear_scale = self.config['controller'].get('front_rear_scale', -1.0)
        delta_rear = front_rear_scale * delta_front
        
        # Apply steering deadband to prevent oscillation near zero
        if abs(delta_front) < self.steering_deadband:
            delta_front = 0.0
            delta_rear = 0.0
        elif abs(delta_front - self.prev_steering[0]) < self.steering_deadband:
            # If change is very small, maintain previous steering
            delta_front = self.prev_steering[0]
            delta_rear = self.prev_steering[1]
        else:
            # Apply exponential smoothing to steering changes
            # Use stronger smoothing for small vehicle speeds
            smoothing_factor = min(0.6, 0.3 + vehicle_speed / 10.0)
            
            if np.any(self.prev_steering != 0):
                delta_front = self.prev_steering[0] + smoothing_factor * (delta_front - self.prev_steering[0])
                delta_rear = self.prev_steering[1] + smoothing_factor * (delta_rear - self.prev_steering[1])
            
        # Store steering angles for next iteration
        self.prev_steering = np.array([delta_front, delta_rear])
            
        return delta_front, delta_rear
    
    def _calculate_motor_voltages(self, v_long, delta_front, vehicle_speed):
        """Calculate motor voltages based on desired velocity.
        
        Args:
            v_long: Longitudinal velocity
            delta_front: Front steering angle
            vehicle_speed: Current vehicle speed
            
        Returns:
            Motor voltages [V_FL, V_FR, V_RL, V_RR]
        """
        # Apply velocity limits
        V_min = self.config['safety']['limits']['velocity']['min']
        V_max = self.config['safety']['limits']['velocity']['max']
        v_base = np.clip(v_long, V_min, V_max)
        
        # Apply deadband to very small velocities to prevent micro-movements
        velocity_deadband = 0.05  # Minimum velocity to actually move
        if abs(v_base) < velocity_deadband:
            v_base = 0
            return np.zeros(4)  # Return zero voltages if below deadband
            
        # For very small steering angles, use straight line motion
        if abs(delta_front) < 0.01:
            # Equal voltages to all wheels for straight motion
            wheel_speeds = np.array([v_base] * 4)
        else:
            # Calculate turn radius
            turn_radius = abs(self.wheelbase / np.tan(delta_front))
            
            # Calculate wheel radii
            inner_radius = max(0.1, turn_radius - self.track_width/2)
            outer_radius = turn_radius + self.track_width/2
            
            # Determine which side is inner/outer based on steering direction
            if delta_front > 0:  # Turning left
                wheel_radius = np.array([
                    inner_radius,  # FL (inner)
                    outer_radius,  # FR (outer)
                    inner_radius,  # RL (inner)
                    outer_radius   # RR (outer)
                ])
            else:  # Turning right
                wheel_radius = np.array([
                    outer_radius,  # FL (outer)
                    inner_radius,  # FR (inner)
                    outer_radius,  # RL (outer)
                    inner_radius   # RR (inner)
                ])
            
            # Calculate wheel speeds
            wheel_speeds = v_base * wheel_radius / turn_radius
            
        # Convert wheel speeds to voltages with basic motor model
        voltage_speed_factor = self.config['robot']['motor']['voltage_speed_factor']
        motor_efficiency = self.config['robot']['motor'].get('efficiency', 0.9)
        
        # Simple conversion from speed to voltage
        motor_voltages = wheel_speeds / (voltage_speed_factor * motor_efficiency)
        
        # Apply voltage limits
        motor_voltages = np.clip(motor_voltages, self.params.u_min, self.params.u_max)
        
        # Apply soft transitions for changes in voltages
        smooth_factor = 0.1  # Smoothing factor (0-1, higher is more responsive)
        motor_voltages = self._smooth_transition(motor_voltages, smooth_factor)
        
        return motor_voltages
        
    def _smooth_transition(self, new_voltages, smooth_factor=0.7):
        """Smooth transitions between voltage values to prevent jerky movements.
        
        Args:
            new_voltages: New voltage values to transition to
            smooth_factor: Smoothing factor (0-1, higher = more responsive)
            
        Returns:
            Smoothed voltage values
        """
        # Only smooth if we have previous values
        if np.any(self.prev_voltages != 0):
            # Calculate difference between previous and new voltages
            voltage_diff = new_voltages - self.prev_voltages
            
            # Apply smoothing based on magnitude of change
            for i in range(len(new_voltages)):
                # Stronger smoothing for small changes to prevent oscillations
                if abs(voltage_diff[i]) < 0.5:
                    local_smooth = smooth_factor * 0.5  # Half the responsiveness for small changes
                else:
                    local_smooth = smooth_factor
                
                # Apply exponential smoothing
                new_voltages[i] = self.prev_voltages[i] + local_smooth * voltage_diff[i]
        
        # Store new voltages for next iteration
        self.prev_voltages = new_voltages.copy()
        
        return new_voltages
