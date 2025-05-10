"""Robot model with front and rear wheel steering and drive capabilities."""

import numpy as np
from typing import Dict, Any, List
from src.models.motor import ElectricMotor

class Robot4WSD:
    """
    Four-wheel steering and drive robot model using ru-racer physics.
    
    This model implements front and rear steering (not independent wheel steering).
    Front wheels share the same steering angle, and rear wheels share the same steering angle.
    Each wheel has its own drive motor.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize robot model with configuration."""
        # State vector: [x, y, θ, vx, vy, omega]
        self.state = np.zeros(6)
        if 'initial' in config:
            self.state[0:3] = config['initial']['state']['position']
            self.state[3:6] = config['initial']['state']['velocity']
            
        # Load physical parameters from config
        self.mass = config['robot']['mass']
        self.inertia = config['robot']['inertia']
        self.wheelbase = config['robot']['wheelbase']  # Distance between front and rear axles
        self.track_width = config['robot']['track_width']  # Distance between left and right wheels
        self.wheel_radius = config['robot']['wheel_radius']
        
        # Steering angles (front and rear)
        self.delta_front = config['initial']['steering']['front']
        self.delta_rear = config['initial']['steering']['rear']
        
        # Initialize motors [FL, FR, RL, RR]
        self.motors = [ElectricMotor(config) for _ in range(4)]
        
        # Control parameters
        self.dt = config['timing']['time_step']
        self.max_velocity = config['safety']['limits']['velocity']['max']
        self.max_angular_velocity = config['safety']['limits']['velocity']['omega_max']
        self.max_steering_angle = config['controller']['mpc']['constraints']['steering']['max']
        self.config = config

    def update(self, action: np.ndarray) -> None:
        """
        Update the robot state based on control inputs.
        
        Args:
            action: Control action [delta_front, delta_rear, V_fl, V_fr, V_rl, V_rr]
                   delta_* are steering angles [rad]
                   V_* are motor voltages [V]
        """
        if self.config['robot']['model'] == 'kinematics':
            self.state = self.kinematics(self.state, action)
        elif self.config['robot']['model'] == 'dynamics':
            self.state = self.dynamics(self.state, action)
        else:
            raise ValueError(f"Unsupported robot model: {self.config['robot']['model']}")

    def kinematics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute robot kinematics (simplified model).
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega]
            action: Control inputs [delta_front, delta_rear, V_fl, V_fr, V_rl, V_rr]
                
        Returns:
            Updated state
        """
        new_state = state.copy()
        
        # Extract control inputs
        delta_front, delta_rear = action[0:2]
        wheel_voltages = action[2:6]
        
        # Apply voltage to motors (simplified in kinematics mode)
        wheel_speeds = []
        for i, motor in enumerate(self.motors):
            # In kinematics mode, we use a simplified relation between voltage and wheel speed
            wheel_speeds.append(wheel_voltages[i] * self.config['robot']['motor'].get('voltage_speed_factor', 0.1))
        
        # Average wheel velocities for simplified model
        v = np.mean(wheel_speeds) * self.wheel_radius
             
        # Calculate vehicle velocities based on steering geometry (bicycle model)
        omega = v * (np.tan(delta_front) - np.tan(delta_rear)) / self.wheelbase
        
        # Local to global velocity transformation
        cos_theta = np.cos(state[2])
        sin_theta = np.sin(state[2])
        vx = v * cos_theta
        vy = v * sin_theta
        
        # Update positions using current velocities
        new_state[0] += vx * self.dt  # x
        new_state[1] += vy * self.dt  # y
        new_state[2] += omega * self.dt  # theta
        
        # Update velocities with some smoothing
        alpha = self.config['controller']['pid'].get('alpha_moving_average', 0.5)  # Smoothing factor
        new_state[3] = (1 - alpha) * state[3] + alpha * vx  # vx
        new_state[4] = (1 - alpha) * state[4] + alpha * vy  # vy
        new_state[5] = (1 - alpha) * state[5] + alpha * omega  # omega
        
        # Store steering angles
        self.delta_front = delta_front
        self.delta_rear = delta_rear
        
        return new_state

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute robot dynamics based on ru-racer physics model.
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega]
            action: Control inputs [delta_front, delta_rear, V_fl, V_fr, V_rl, V_rr]
                
        Returns:
            Updated state
        """
        # Extract current state
        x, y, theta, vx, vy, omega = state
        
        # Extract control inputs
        delta_front, delta_rear = action[0:2]
        wheel_voltages = action[2:6]
        
        # Store steering angles
        self.delta_front = delta_front
        self.delta_rear = delta_rear
        
        # Calculate slip angles for each wheel based on ru-racer Bike.m
        slip_angles = self._compute_slip_angles(state, delta_front, delta_rear)
        
        # Calculate wheel velocities based on vehicle motion
        wheel_velocities = self._compute_wheel_velocities(state, delta_front, delta_rear)
        
        # Calculate normal forces (static distribution for simplicity)
        normal_forces = self._compute_normal_forces()
        
        # Update each motor and get tire forces
        F_x = np.zeros(4)
        F_y = np.zeros(4)
        
        for i in range(4):
            F_x[i], F_y[i] = self.motors[i].update(
                wheel_voltages[i],
                wheel_velocities[i], 
                slip_angles[i], 
                normal_forces[i],
                self.dt
            )
        
        # Transform forces to vehicle coordinate system
        F_x_vehicle = np.zeros(4)
        F_y_vehicle = np.zeros(4)
        
        # Front wheels
        for i in range(2):  # FL, FR
            F_x_vehicle[i] = F_x[i] * np.cos(delta_front) - F_y[i] * np.sin(delta_front)
            F_y_vehicle[i] = F_x[i] * np.sin(delta_front) + F_y[i] * np.cos(delta_front)
            
        # Rear wheels
        for i in range(2, 4):  # RL, RR
            F_x_vehicle[i] = F_x[i] * np.cos(delta_rear) - F_y[i] * np.sin(delta_rear)
            F_y_vehicle[i] = F_x[i] * np.sin(delta_rear) + F_y[i] * np.cos(delta_rear)
        
        # Sum forces in each direction
        F_x_total = np.sum(F_x_vehicle)
        F_y_total = np.sum(F_y_vehicle)
        
        # Calculate moment around CoG
        half_track = self.track_width / 2
        half_wheelbase = self.wheelbase / 2
        
        # Calculate wheel positions
        wheel_positions = [
            [half_wheelbase, half_track],    # FL
            [half_wheelbase, -half_track],   # FR
            [-half_wheelbase, half_track],   # RL
            [-half_wheelbase, -half_track]   # RR
        ]
        
        # Calculate moment
        M_z = 0.0
        for i in range(4):
            M_z += wheel_positions[i][0] * F_y_vehicle[i] - wheel_positions[i][1] * F_x_vehicle[i]
        
        # Apply Newton's second law (F = ma) with centripetal effects
        a_x = F_x_total / self.mass + vy * omega
        a_y = F_y_total / self.mass - vx * omega
        alpha = (M_z / self.inertia)
        
        # Integrate accelerations using Euler method
        new_vx = vx + a_x * self.dt
        new_vy = vy + a_y * self.dt
        new_omega = omega + alpha * self.dt
        
        # Apply velocity limits
        new_vx = np.clip(new_vx, -self.max_velocity, self.max_velocity)
        new_vy = np.clip(new_vy, -self.max_velocity, self.max_velocity)
        # new_omega = np.clip(new_omega, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Update positions (body frame to global frame)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        new_x = x + (new_vx * cos_theta - new_vy * sin_theta) * self.dt
        new_y = y + (new_vx * sin_theta + new_vy * cos_theta) * self.dt
        new_theta = theta + new_omega * self.dt
        
        # Normalize theta to [-π, π]
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
        
        return np.array([new_x, new_y, new_theta, new_vx, new_vy, new_omega])
        
    def _compute_slip_angles(self, state: np.ndarray, delta_front: float, delta_rear: float) -> List[float]:
        """
        Calculate slip angles for each wheel based on ru-racer Bike.m.
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega]
            delta_front: Front steering angle [rad]
            delta_rear: Rear steering angle [rad]
            
        Returns:
            List of slip angles [FL, FR, RL, RR]
        """
        vx, vy, omega = state[3:6]
        

        # Front slip angle (from ru-racer Bike.m)
        slip_front = np.arctan2(
            (vy + self.wheelbase/2 * omega) * np.cos(delta_front) - vx * np.sin(delta_front),
            vx * np.cos(delta_front) + (vy + self.wheelbase/2 * omega) * np.sin(delta_front)
        )
        
        # Rear slip angle
        slip_rear = np.arctan2(vy - self.wheelbase/2 * omega, vx)

        # Return slip angles for all wheels [FL, FR, RL, RR]
        return [slip_front, slip_front, slip_rear, slip_rear]
    
    def _compute_wheel_velocities(self, state: np.ndarray, delta_front: float, delta_rear: float) -> List[float]:
        """
        Calculate wheel velocities at tire contact point.
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega]
            delta_front: Front steering angle [rad]
            delta_rear: Rear steering angle [rad]
            
        Returns:
            List of wheel velocities [FL, FR, RL, RR]
        """
        vx, vy, omega = state[3:6]
        
        # Calculate front wheel velocities in wheel coordinate system
        v_front = np.sqrt(vx**2 + (vy + omega * self.wheelbase/2)**2)
        front_direction = np.arctan2(vy + omega * self.wheelbase/2, vx)
        v_front_x = v_front * np.cos(front_direction - delta_front)
        
        # Rear wheel velocities in wheel coordinate system
        v_rear = np.sqrt(vx**2 + (vy - omega * self.wheelbase/2)**2)
        rear_direction = np.arctan2(vy - omega * self.wheelbase/2, vx)
        v_rear_x = v_rear * np.cos(rear_direction - delta_rear)
        
        # Return longitudinal velocity at each wheel [FL, FR, RL, RR]
        return [v_front_x, v_front_x, v_rear_x, v_rear_x]
    
    def _compute_normal_forces(self) -> List[float]:
        """
        Calculate normal forces on each wheel (static distribution).
        
        Returns:
            List of normal forces [FL, FR, RL, RR]
        """
        g = 9.81  # Gravity constant
        
        # Static weight distribution (forward weight bias)
        weight_bias_front = self.config.get('robot', {}).get('weight_bias_front', 0.5)
        F_normal_front = self.mass * g * weight_bias_front / 2  # Divide by 2 for left/right
        F_normal_rear = self.mass * g * (1 - weight_bias_front) / 2  # Divide by 2 for left/right
        
        # Return normal forces [FL, FR, RL, RR]
        return [F_normal_front, F_normal_front, F_normal_rear, F_normal_rear]
