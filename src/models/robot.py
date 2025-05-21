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
        self.state_history = {0.0: self.state.copy()}

    def update(self, action: np.ndarray) -> None:
        """
        Update the robot state based on control inputs.
        
        Args:
            action: Control action [delta_front, delta_rear, V_fl, V_fr, V_rl, V_rr]
                   delta_* are steering angles [rad]
                   V_* are motor voltages [V]
        """
        # Normalize theta before every update to prevent drift
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
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
        new_state = state
        
        # Extract control inputs
        delta_front, delta_rear = action[0:2]
        wheel_voltages = action[2:6]
        
        # Apply voltage to motors (simplified but more accurately matching dynamics model)
        voltage_speed_factor = self.config['robot']['motor'].get('voltage_speed_factor', 0.1)
        motor_efficiency = self.config['robot']['motor'].get('efficiency', 0.9)
        gear_ratio = self.config['robot']['motor'].get('gear_ratio', 15.0)
        wheel_speeds = []

        for i, motor in enumerate(self.motors):
            # Apply a more realistic voltage-to-speed conversion that better matches dynamics mode
            effective_voltage = wheel_voltages[i] * motor_efficiency
            wheel_speeds.append(effective_voltage)

        # Average wheel velocities for simplified model
        v = np.mean(wheel_speeds) * self.wheel_radius / (voltage_speed_factor * gear_ratio)

        # IMPORTANT: First normalize theta before calculating anything else
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        
        # Calculate vehicle velocities based on steering geometry (bicycle model)
        omega = v * (np.tan(delta_front) - np.tan(delta_rear)) / self.wheelbase        
    
        # Local to global velocity transformation
        cos_theta = np.cos(new_state[2])
        sin_theta = np.sin(new_state[2])
        vx = v * cos_theta
        vy = v * sin_theta
        
        # Apply physical limits on body-frame velocities
        vx = np.clip(vx, -self.max_velocity, self.max_velocity)
        vy = np.clip(vy, -self.max_velocity, self.max_velocity)
        omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)

        # Update positions using current velocities
        new_state[0] += vx * self.dt  # x
        new_state[1] += vy * self.dt  # y
        new_state[2] += omega * self.dt  # theta
        
        # Update velocities with some smoothing
        alpha = self.config['controller']['pid'].get('alpha_moving_average', 0.5)  # Smoothing factor
        
        # Apply velocity smoothing to reduce steady state error
        # This allows the controller to gradually correct errors rather than immediately overriding
        target_vx = vx
        target_vy = vy
        target_omega = omega
        
        # Only apply blending for non-zero targets to avoid "sticking" at zero
        if np.abs(target_vx) > 0.001 or np.abs(target_vy) > 0.001 or np.abs(target_omega) > 0.001:
            # Blend current velocity with desired velocity using smoothing factor
            new_state[3] = (1 - alpha) * new_state[3] + alpha * target_vx
            new_state[4] = (1 - alpha) * new_state[4] + alpha * target_vy
            new_state[5] = (1 - alpha) * new_state[5] + alpha * target_omega
        else:
            # For zero targets, allow full deceleration to zero
            new_state[3] = target_vx
            new_state[4] = target_vy
            new_state[5] = target_omega
        
        # Store steering angles
        self.delta_front = delta_front
        self.delta_rear = delta_rear
        
        # Final normalization of theta to ensure it stays within [-π, π]
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        # print("new_state:", new_state, "action:", action)

        return new_state

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute robot dynamics based on physics model.
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega]
            action: Control inputs [delta_front, delta_rear, V_fl, V_fr, V_rl, V_rr]
                
        Returns:
            Updated state
        """
        new_state = state.copy()
        
        # Extract current state
        x, y, theta = state[0:3]  # Global position and orientation
        vx, vy, omega = state[3:6]  # Global velocities
        
        # Extract control inputs
        delta_front, delta_rear = action[0:2]
        wheel_voltages = action[2:6]
        
        # Store steering angles for future reference
        self.delta_front = delta_front
        self.delta_rear = delta_rear
        
        # IMPORTANT: First normalize theta before calculating anything else
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        
        # Transform global velocities to body frame velocities
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Create rotation matrix from global to body frame
        # Standard transformation for 2D rotation from global to body frame
        R_global_to_body = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])
        
        # Apply rotation to get body frame velocities
        velocities_global = np.array([vx, vy])
        vx_body, vy_body = R_global_to_body @ velocities_global
        
        # Step 1: Calculate slip angles and wheel velocities using body-frame velocities
        slip_angles = self._compute_slip_angles(np.array([x, y, theta, vx_body, vy_body, omega]), delta_front, delta_rear)
        wheel_velocities = self._compute_wheel_velocities(np.array([x, y, theta, vx_body, vy_body, omega]), delta_front, delta_rear)
        normal_forces = self._compute_normal_forces()
        
        # Apply voltage scaling to match kinematics model velocities
        voltage_speed_factor = self.config['robot']['motor'].get('voltage_speed_factor', 0.1)
        motor_efficiency = self.config['robot']['motor'].get('efficiency', 0.9)
        gear_ratio = self.config['robot']['motor'].get('gear_ratio', 15.0)
        
        # Scale wheel voltages to accurately match kinematics behavior
        scaled_wheel_voltages = []
        for i in range(4):
            # Apply efficiency scaling (same as in kinematics)
            effective_voltage = wheel_voltages[i] * motor_efficiency
            # Apply correct scaling factor to match the kinematics model
            wheel_factor = self.wheel_radius / (voltage_speed_factor * gear_ratio)
            scaled_wheel_voltages.append(effective_voltage * wheel_factor)
        
        # Step 2: Calculate tire forces from motor dynamics with scaled voltages
        F_x = np.zeros(4)  # Longitudinal forces at each wheel [FL, FR, RL, RR]
        F_y = np.zeros(4)  # Lateral forces at each wheel [FL, FR, RL, RR]
        
        for i, motor in enumerate(self.motors):
            F_x[i], F_y[i] = motor.update(
                scaled_wheel_voltages[i],
                wheel_velocities[i], 
                slip_angles[i], 
                normal_forces[i],
                self.dt
            )
        
        # Step 3: Transform wheel forces to vehicle body frame
        F_x_vehicle = np.zeros(4)
        F_y_vehicle = np.zeros(4)
        
        # Front wheels transformation (rotate by steering angle)
        for i in range(2):  # FL, FR
            F_x_vehicle[i] = F_x[i] * np.cos(delta_front) - F_y[i] * np.sin(delta_front)
            F_y_vehicle[i] = F_x[i] * np.sin(delta_front) + F_y[i] * np.cos(delta_front)
            
        # Rear wheels transformation (rotate by steering angle)
        for i in range(2, 4):  # RL, RR
            F_x_vehicle[i] = F_x[i] * np.cos(delta_rear) - F_y[i] * np.sin(delta_rear)
            F_y_vehicle[i] = F_x[i] * np.sin(delta_rear) + F_y[i] * np.cos(delta_rear)
        
        # Step 4: Sum forces for total force on vehicle body
        F_x_total = np.sum(F_x_vehicle)
        F_y_total = np.sum(F_y_vehicle)
        
        # Step 5: Calculate moment (torque) around CoG
        half_track = self.track_width / 2
        half_wheelbase = self.wheelbase / 2
        
        # Wheel positions relative to CoG [x, y]
        wheel_positions = [
            [half_wheelbase, half_track],    # FL
            [half_wheelbase, -half_track],   # FR
            [-half_wheelbase, half_track],   # RL
            [-half_wheelbase, -half_track]   # RR
        ]
        
        # Calculate moment using correct cross product formula (r × F)z = rx*Fy - ry*Fx
        M_z = 0.0
        for i in range(4):
            # For z-component of torque in right-hand coordinate system
            M_z += wheel_positions[i][0] * F_y_vehicle[i] - wheel_positions[i][1] * F_x_vehicle[i]
        
        # Step 6: Apply Newton's second law with proper rigid body dynamics
        # Linear acceleration (F = ma) with centripetal effects for planar motion
        # Note: centripetal terms account for rotating reference frame effects
        a_x_body = F_x_total / self.mass + vy_body * omega  # X acceleration in body frame
        a_y_body = F_y_total / self.mass - vx_body * omega  # Y acceleration in body frame
        
        # Angular acceleration (T = I*alpha)
        alpha = M_z / self.inertia  # Angular acceleration around Z-axis
        
        # Step 7: Integrate accelerations to get new velocities (body frame)
        # First-order Euler integration
        new_vx_body = vx_body + a_x_body * self.dt
        new_vy_body = vy_body + a_y_body * self.dt
        new_omega = omega + alpha * self.dt
        
        # Apply physical limits on body-frame velocities
        new_vx_body = np.clip(new_vx_body, -self.max_velocity, self.max_velocity)
        new_vy_body = np.clip(new_vy_body, -self.max_velocity, self.max_velocity)
        new_omega = np.clip(new_omega, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Step 8: Update orientation (theta)
        new_theta = theta + new_omega * self.dt
        
        # Normalize to [-π, π] for numerical stability
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
        
        # Step 9: Transform body frame velocities back to global frame
        # Prioritize forward motion to prevent sliding
        # This matches the kinematics model where the car primarily moves in its forward direction
        v = new_vx_body  # Use forward velocity as primary speed

        # Apply direct transformation like in kinematics method
        # In kinematics: vx = v * cos(theta), vy = v * sin(theta)
        cos_new_theta = np.cos(new_theta)
        sin_new_theta = np.sin(new_theta)
        new_vx_global = v * cos_new_theta
        new_vy_global = v * sin_new_theta
        
        # Step 10: Update global position using global velocities
        new_x = x + new_vx_global * self.dt
        new_y = y + new_vy_global * self.dt
        
        # Update velocities in state with some smoothing (match kinematics method)
        alpha = self.config['controller']['pid'].get('alpha_moving_average', 0.5)  # Smoothing factor
        
        # Apply velocity smoothing to reduce steady state error
        # This allows the controller to gradually correct errors rather than immediately overriding
        target_vx = new_vx_global
        target_vy = new_vy_global
        target_omega = new_omega
        
        # Only apply blending for non-zero targets to avoid "sticking" at zero
        if np.abs(target_vx) > 0.001 or np.abs(target_vy) > 0.001 or np.abs(target_omega) > 0.001:
            # Blend current velocity with desired velocity using smoothing factor
            new_state[3] = (1 - alpha) * new_state[3] + alpha * target_vx
            new_state[4] = (1 - alpha) * new_state[4] + alpha * target_vy
            new_state[5] = (1 - alpha) * new_state[5] + alpha * target_omega
        else:
            # For zero targets, allow full deceleration to zero
            new_state[3] = target_vx
            new_state[4] = target_vy
            new_state[5] = target_omega
        
        # Update positions and orientation
        new_state[0] = new_x
        new_state[1] = new_y
        new_state[2] = new_theta
        
        # Final normalization of theta to ensure it stays within [-π, π]
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
        
        # Return updated state
        return new_state

    def predict(self, state, action, dt: float):
        """
        Predict the next state based on current state and control input.
        Used by MPC controllers for state prediction over the prediction horizon.
        This method handles both numerical numpy arrays and symbolic CasADi variables.
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega] (could be symbolic for MPC)
            action: Control input [delta_front, delta_rear, V_FL, V_FR, V_RL, V_RR] (could be symbolic for MPC)
            dt: Time step for prediction
            
        Returns:
            Predicted next state
        """
        # Import casadi to check type
        import casadi as ca
        
        # Extract state components
        x, y, theta = state[0], state[1], state[2]
        vx, vy, omega = state[3], state[4], state[5]
        
        # Extract control inputs
        delta_front, delta_rear = action[0], action[1]
        
        # For symbolic expression, we need to explicitly handle the wheel voltages
        # For MPC with casadi, action might be a symbolic expression
        if isinstance(action, ca.MX):
            v_fl = action[2]
            v_fr = action[3]
            v_rl = action[4]
            v_rr = action[5]
            # Calculate mean voltage (simplified)
            mean_voltage = (v_fl + v_fr + v_rl + v_rr) / 4.0
        else:
            # For numpy arrays
            wheel_voltages = action[2:6]
            mean_voltage = sum(wheel_voltages) / 4.0
        
        # Get motor parameters (using the same defaults as dynamics method)
        voltage_speed_factor = self.config['robot']['motor'].get('voltage_speed_factor', 0.1)
        motor_efficiency = self.config['robot']['motor'].get('efficiency', 0.9)
        gear_ratio = self.config['robot']['motor'].get('gear_ratio', 15.0)
        
        # Apply efficiency
        effective_voltage = mean_voltage * motor_efficiency
        
        # Calculate velocity from voltage (simplified model)
        v = effective_voltage * self.wheel_radius / (voltage_speed_factor * gear_ratio)
        
        # Calculate angular velocity based on steering (bicycle model)
        if isinstance(delta_front, ca.MX):
            # Use CasADi functions
            omega_pred = v * (ca.tan(delta_front) - ca.tan(delta_rear)) / self.wheelbase
            
            # Transform to global frame
            cos_theta = ca.cos(theta)
            sin_theta = ca.sin(theta)
            
            # Global velocities
            vx_pred = v * cos_theta
            vy_pred = v * sin_theta
            
            # Update position and orientation
            x_next = x + vx_pred * dt
            y_next = y + vy_pred * dt
            theta_next = theta + omega_pred * dt
            
            # Normalize theta
            theta_next = ca.atan2(ca.sin(theta_next), ca.cos(theta_next))
            
            # Create next state vector
            next_state = ca.vertcat(x_next, y_next, theta_next, vx_pred, vy_pred, omega_pred)
        else:
            # Use NumPy functions
            omega_pred = v * (np.tan(delta_front) - np.tan(delta_rear)) / self.wheelbase
            
            # Transform to global frame
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Global velocities
            vx_pred = v * cos_theta
            vy_pred = v * sin_theta
            
            # Update position and orientation
            x_next = x + vx_pred * dt
            y_next = y + vy_pred * dt
            theta_next = theta + omega_pred * dt
            
            # Normalize theta
            theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
            
            # Create next state vector - for numpy arrays
            next_state = np.array([x_next, y_next, theta_next, vx_pred, vy_pred, omega_pred])
            
        return next_state
        
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
        # Extract body-frame velocities and angular velocity
        vx, vy, omega = state[3:6]
        
        # IMPORTANT: Ensure velocities are in body frame
        # The logic below assumes vx, vy are already in the body frame
        
        # Step 1: Calculate velocities at wheel centers
        # Front wheels velocity components (body frame)
        vx_front = vx  # Same longitudinal velocity as CoG
        vy_front = vy + self.wheelbase/2 * omega  # Add rotational component
        
        # Rear wheels velocity components (body frame)
        vx_rear = vx  # Same longitudinal velocity as CoG
        vy_rear = vy - self.wheelbase/2 * omega  # Subtract rotational component
        
        # Step 2: Transform to wheel-aligned coordinates (account for steering)
        # Front slip angle: Project wheel center velocity onto wheel coordinate system
        slip_front = np.arctan2(
            (vy_front) * np.cos(delta_front) - vx_front * np.sin(delta_front),
            vx_front * np.cos(delta_front) + (vy_front) * np.sin(delta_front)
        )
        
        # Rear slip angle: Project wheel center velocity onto wheel coordinate system
        slip_rear = np.arctan2(vy_rear, vx_rear)
        
        # Step 3: Handle special cases (numerical stability)
        # If vehicle is not moving, slip angle is zero
        if np.sqrt(vx**2 + vy**2) < 0.001:
            slip_front = 0.0
            slip_rear = 0.0
        
        # Step 4: Return slip angles for all wheels [FL, FR, RL, RR]
        # Same slip angle for left and right wheels on same axle
        return [slip_front, slip_front, slip_rear, slip_rear]

    def _compute_wheel_velocities(self, state: np.ndarray, delta_front: float, delta_rear: float) -> List[float]:
        """
        Calculate wheel velocities at tire contact point, following a similar approach as the kinematics method.
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega]
            delta_front: Front steering angle [rad]
            delta_rear: Rear steering angle [rad]
            
        Returns:
            List of wheel velocities [FL, FR, RL, RR]
        """
        # Step 1: Extract body-frame velocities and angular velocity
        vx, vy, omega = state[3:6]
        
        # IMPORTANT: First ensure velocities are in body frame
        # The logic below assumes vx, vy are already in the body frame
        
        # Step 2: Calculate wheel center velocities relative to CoG
        # Front wheels - add rotational component
        vx_front = vx  # Same longitudinal velocity as CoG
        vy_front = vy + omega * self.wheelbase/2  # Add rotational component
        
        # Rear wheels - subtract rotational component
        vx_rear = vx  # Same longitudinal velocity as CoG
        vy_rear = vy - omega * self.wheelbase/2  # Subtract rotational component
        
        # Step 3: Calculate speed (magnitude) and direction in body frame
        # Front wheels
        v_front = np.sqrt(vx_front**2 + vy_front**2)  # Speed magnitude
        front_direction = np.arctan2(vy_front, vx_front)  # Direction angle
        
        # Rear wheels
        v_rear = np.sqrt(vx_rear**2 + vy_rear**2)  # Speed magnitude
        rear_direction = np.arctan2(vy_rear, vx_rear)  # Direction angle
        
        # Step 4: Transform speeds to wheel's local coordinate system (account for steering)
        # Front wheels - rotate velocity vector by steering angle
        v_front_x = v_front * np.cos(front_direction - delta_front)
        
        # Rear wheels - rotate velocity vector by steering angle
        v_rear_x = v_rear * np.cos(rear_direction - delta_rear)
        
        # Step 5: Handle edge cases for numerical stability
        # When velocity is very small, avoid numerical issues
        if v_front < 0.001:
            v_front_x = 0.0
        
        if v_rear < 0.001:
            v_rear_x = 0.0
        
        # Step 6: Return longitudinal velocity at each wheel [FL, FR, RL, RR]
        # Same value for left and right wheels on same axle
        return [v_front_x, v_front_x, v_rear_x, v_rear_x]

    def _compute_normal_forces(self) -> List[float]:
        """
        Calculate normal forces on each wheel (static distribution).
        
        Based on weight bias and mass properties, this method computes the normal forces
        applied at each tire contact point, accounting for weight distribution.
        
        Returns:
            List of normal forces [FL, FR, RL, RR]
        """
        # Step 1: Define constants
        g = 9.81  # Gravity constant [m/s^2]
        
        # Step 2: Get vehicle weight distribution from config
        # Static weight distribution (forward weight bias)
        weight_bias_front = self.config.get('robot', {}).get('weight_bias_front', 0.5)
        
        # Step 3: Calculate forces for front and rear axles
        # Total weight distributed to front axle
        front_axle_weight = self.mass * g * weight_bias_front
        # Total weight distributed to rear axle
        rear_axle_weight = self.mass * g * (1 - weight_bias_front)
        
        # Step 4: Distribute axle weights to individual wheels (symmetrically)
        # Divide front axle weight equally between left and right
        F_normal_front = front_axle_weight / 2  
        # Divide rear axle weight equally between left and right
        F_normal_rear = rear_axle_weight / 2
        
        # Step 5: Return normal forces for all four wheels [FL, FR, RL, RR]
        return [F_normal_front, F_normal_front, F_normal_rear, F_normal_rear]
