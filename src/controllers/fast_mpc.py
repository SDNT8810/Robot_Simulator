"""Fast Model Predictive Control implementation based on gradient descent optimization.

This implementation is inspired by the MATLAB code in docs/ru-racer/Uturn.m
and provides a faster alternative to the CasADi-based MPC implementation.
"""

import numpy as np
import time
from dataclasses import dataclass
from src.models.robot import Robot4WSD

@dataclass
class MPCParams:
    """MPC parameters from config"""
    Hp: int  # Prediction horizon
    dt: float  # Time step
    Q1: np.ndarray  # Position tracking weights
    Q2: np.ndarray  # Velocity tracking weights
    R: np.ndarray  # Control input weights
    u_min: np.ndarray  # Input lower bounds
    u_max: np.ndarray  # Input upper bounds
    du_min: np.ndarray  # Input rate lower bounds
    du_max: np.ndarray  # Input rate upper bounds


class FastMPC:
    """Fast MPC controller using gradient-based optimization.
    
    This implementation avoids using CasADi and instead uses a custom gradient-based
    optimization approach similar to the one in the MATLAB code from docs/ru-racer/Uturn.m.
    """
    
    def __init__(self, config: dict):
        """Initialize MPC controller.
        
        Args:
            config: Configuration
        """
        # Load configuration
        self.config = config

        # Set up MPC parameters
        q1_list = [float(val) for val in self.config['controller']['mpc']['weights']['Q1']]
        q2_list = [float(val) for val in self.config['controller']['mpc']['weights']['Q2']]
        r_list = [float(val) for val in self.config['controller']['mpc']['weights']['R']]
        
        # Convert steering and voltage constraints to float arrays
        steering_min = [float(val) for val in self.config['controller']['mpc']['constraints']['steering']['min']]
        steering_max = [float(val) for val in self.config['controller']['mpc']['constraints']['steering']['max']]
        steering_rate_min = [float(val) for val in self.config['controller']['mpc']['constraints']['steering']['rate_min']]
        steering_rate_max = [float(val) for val in self.config['controller']['mpc']['constraints']['steering']['rate_max']]
        
        voltage_min = [float(val) for val in self.config['controller']['mpc']['constraints']['voltage']['min']]
        voltage_max = [float(val) for val in self.config['controller']['mpc']['constraints']['voltage']['max']]
        voltage_rate_min = [float(val) for val in self.config['controller']['mpc']['constraints']['voltage']['rate_min']]
        voltage_rate_max = [float(val) for val in self.config['controller']['mpc']['constraints']['voltage']['rate_max']]
        
        # Create MPCParams object with explicit float conversions
        self.params = MPCParams(
            Hp=int(self.config['controller']['mpc']['prediction_horizon']),
            dt=float(self.config['controller']['mpc']['time_step']),
            Q1=np.diag(np.array(q1_list, dtype=float)),
            Q2=np.diag(np.array(q2_list, dtype=float)),
            R=np.diag(np.array(r_list, dtype=float)),
            u_min=np.array(steering_min + voltage_min, dtype=float),
            u_max=np.array(steering_max + voltage_max, dtype=float),
            du_min=np.array(steering_rate_min + voltage_rate_min, dtype=float),
            du_max=np.array(steering_rate_max + voltage_rate_max, dtype=float)
        )
        
        # Initialize robot model for prediction
        self.robot_model = Robot4WSD(config)
        
        # Cache for last control input (for rate limiting)
        self.last_u = np.zeros(len(self.params.u_min), dtype=float)
        
        # Store last prediction for visualization
        self.last_solution = None
        self.predicted_states = None
        
        # Add integral action for tracking moving targets - only for position (x,y)
        self.integral_error = np.zeros(2, dtype=float)  # [x, y] position error integral only
        # Weight for integral action - increased for faster error correction
        self.integral_weight = np.array([0.2, 0.2], dtype=float)
        # Anti-windup limit
        self.integral_limit = 3.0
        
        # Optimization parameters
        if 'optimization' in config:
            self.max_iterations = int(config['optimization'].get('max_iterations', 20))
            self.learning_rate = float(config['optimization'].get('learning_rate', 1.0))
            self.min_accuracy = float(config['optimization'].get('min_accuracy', 1e-3))
            self.max_time = float(config['optimization'].get('max_time', 0.02))
        else:
            self.max_iterations = 20
            self.learning_rate = 1.0
            self.min_accuracy = 1e-3
            self.max_time = 0.02
        
    def action(self, state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        """Compute control action using MPC.
        
        Args:
            state: Current robot state [x, y, θ, vx, vy, omega]
            desired_state: Desired robot state [x, y, θ, vx, vy, omega]
                
        Returns:
            Control action [δ_front, δ_rear, V_FL, V_FR, V_RL, V_RR]
            where δ are steering angles in radians and V are motor voltages.
        """
        # Convert inputs to float arrays to ensure type compatibility
        state_float = np.array(state, dtype=float)
        desired_state_float = np.array(desired_state, dtype=float)
        
        # Calculate current position error for logging/debugging
        position_error = np.linalg.norm(desired_state_float[0:2] - state_float[0:2])
        
        # Update integral error for position ONLY
        pos_error_vec = desired_state_float[0:2] - state_float[0:2]
        
        # Update integral error with anti-windup
        self.integral_error += pos_error_vec * self.params.dt
        
        # If error is decreasing, we can reduce integral term to avoid overshoot
        if position_error < 0.1:  # Small threshold to detect when we're close
            self.integral_error *= 0.95  # Gradual reduction
        
        # Apply integral limits to prevent windup
        for i in range(2):
            self.integral_error[i] = np.clip(self.integral_error[i], -self.integral_limit, self.integral_limit)
        
        # Modify target by adding integral correction to position
        modified_desired_state = desired_state_float.copy()
        modified_desired_state[0:2] += self.integral_weight * self.integral_error
        
        # Apply MPC to get optimal control action
        u_opt = self.solve_fast_mpc(state_float, modified_desired_state)
        
        # Save current control for next iteration (for rate limiting)
        self.last_u = u_opt.copy()
        
        return u_opt
    
    def solve_fast_mpc(self, state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        """Solve the MPC optimization problem using gradient-based optimization.
        
        Args:
            state: Current state of the robot
            desired_state: Desired state to track
            
        Returns:
            Optimal control action
        """
        # Extract MPC parameters
        Hp = self.params.Hp
        dt = self.params.dt
        u_min = self.params.u_min
        u_max = self.params.u_max
        
        # Calculate position error and magnitude for use throughout the function
        pos_error = desired_state[0:2] - state[0:2]
        error_magnitude = float(np.linalg.norm(pos_error))
        
        # For a true MPC, we need to optimize a sequence of controls
        # Let's create a control horizon (Hc) - shorter than prediction horizon
        # for computational efficiency
        Hc = min(5, Hp)  # Control horizon, max 5 steps
        
        # Create an initial sequence of control inputs
        # Initialize with the last control repeated for Hc steps
        u_sequence = np.tile(self.last_u.copy(), (Hc, 1))
        
        # If we're far from the goal and the initial control is zero, set a good initial guess
        if error_magnitude > 1.0 and np.all(np.abs(u_sequence) < 0.01):
            # Calculate desired heading
            heading = np.arctan2(pos_error[1], pos_error[0])
            
            # Create a reasonable initial guess - front wheel steers toward goal
            initial_steering = np.clip(heading, -0.5, 0.5)
            
            # Set initial voltage based on distance
            initial_voltage = min(8.0, float(error_magnitude))
            
            # For each step in the control horizon
            for i in range(Hc):
                # Set steering
                u_sequence[i, 0] = initial_steering  # Front steering
                u_sequence[i, 1] = 0.0  # Rear steering stays neutral
                
                # Set voltages if they're part of the optimization variables
                if u_sequence.shape[1] > 2:  # If we have voltage control
                    u_sequence[i, 2:6] = initial_voltage  # Set voltages for all motors
        
        # Perform optimization on the entire control sequence
        u_opt_sequence = self.optimize_sequence(state, desired_state, u_sequence, dt, Hp, Hc)
        
        # Store the optimal control sequence for visualization and future use
        self.last_solution = [u_opt_sequence[i] for i in range(Hc)]
        
        # Apply receding horizon principle: only use the first control action
        u_opt = u_opt_sequence[0].copy()
        
        # Apply input constraints and ensure correct type
        u_opt = np.clip(u_opt, u_min, u_max).astype(float)
        
        # Ensure we have the right shape (6,) for control action
        if len(u_opt) == 2:  # If we only have steering angles
            # Add default voltages for the 4 motors
            
            # Scale voltage based on distance to goal and heading alignment
            base_voltage = 8.0  # Base voltage value when far from goal
            voltage = base_voltage * min(1.0, float(error_magnitude / 5.0))  # Scale down when closer
            
            # Calculate desired heading 
            heading = np.arctan2(pos_error[1], pos_error[0])
            
            # Set steering to aim toward goal if optimization didn't already
            if abs(u_opt[0]) < 0.01:  # If front steering is near zero
                u_opt[0] = np.clip(heading, -0.5, 0.5)  # Front steering
            
            # Apply the same voltage to all wheels for initial movement
            u_opt = np.concatenate([u_opt, np.array([voltage, voltage, voltage, voltage], dtype=float)])
        
        # Explicitly ensure non-zero motor voltages when we have a significant position error
        if error_magnitude > 0.1 and len(u_opt) >= 6:
            # Ensure at least some of the motor voltages are non-zero
            if np.all(np.abs(u_opt[2:6]) < 0.1):  # If all motor voltages are near zero
                # Apply default voltages based on distance
                voltage_magnitude = min(8.0, float(error_magnitude))  # Cap at 8V
                u_opt[2:6] = np.array([voltage_magnitude, voltage_magnitude, 
                                      voltage_magnitude, voltage_magnitude])
                
        return u_opt
    
    def optimize_sequence(self, state: np.ndarray, desired_state: np.ndarray, 
                       u_sequence: np.ndarray, dt: float, Hp: int, Hc: int) -> np.ndarray:
        """Perform gradient-based optimization on a sequence of control inputs.
        
        Args:
            state: Current state
            desired_state: Desired state
            u_sequence: Initial sequence of control inputs (Hc x nu)
            dt: Time step
            Hp: Prediction horizon
            Hc: Control horizon
            
        Returns:
            Optimized sequence of control inputs
        """
        # Initialize optimization
        u_current = u_sequence.copy()
        iteration = 0
        grad_norm = float('inf')
        learning_rate = float(self.learning_rate * 1.5)  # Increased learning rate
        best_u = u_current.copy()
        best_cost = float('inf')
        
        # Start timing for optimization
        start_time = time.time()
        
        # Main optimization loop
        while (iteration < self.max_iterations and 
               grad_norm > self.min_accuracy and 
               time.time() - start_time < self.max_time):
            # Compute cost function for the entire sequence
            fx = self.compute_sequence_cost(desired_state, state, u_current, dt, Hp, Hc)
            
            # Track best solution found so far
            if fx < best_cost:
                best_cost = fx
                best_u = u_current.copy()
            
            # Compute gradient for the entire sequence
            grad = self.compute_sequence_gradient(u_current, fx, desired_state, state, dt, Hp, Hc)
            grad_norm = np.linalg.norm(grad)
            
            # Simple gradient descent step
            u_new = u_current - learning_rate * grad
            
            # Apply input constraints to all steps in the sequence
            for i in range(Hc):
                u_new[i] = np.clip(u_new[i], self.params.u_min, self.params.u_max)
            
            # Compute cost with new control sequence
            fx_new = self.compute_sequence_cost(desired_state, state, u_new, dt, Hp, Hc)
            
            # Update control sequence if cost improved
            if fx_new < fx:
                u_current = u_new.copy()
                learning_rate = learning_rate * 1.2  # Increase learning rate if improved
            else:
                learning_rate = learning_rate * 0.5  # Decrease learning rate if not improved
            
            iteration += 1
        
        # Return the best solution found during optimization
        return best_u
        
    def compute_sequence_cost(self, desired_state: np.ndarray, initial_state: np.ndarray, 
                              u_sequence: np.ndarray, dt: float, Hp: int, Hc: int) -> float:
        """Compute cost function for a sequence of control inputs over the horizon.
        
        Args:
            desired_state: Desired state trajectory
            initial_state: Initial state
            u_sequence: Sequence of control inputs (Hc x nu)
            dt: Time step
            Hp: Prediction horizon
            Hc: Control horizon
            
        Returns:
            Cost value
        """
        # Initialize cost
        cost = 0.0
        
        # Initialize state trajectory
        z = np.zeros((6, Hp+1), dtype=float)
        z[:, 0] = initial_state.copy()
        
        # Position weight is high to focus on position tracking
        position_weight = 10.0  # Higher weight for position tracking
        
        # Terminal cost weight
        terminal_weight = 5.0  # Higher weight for terminal state
        
        # Input regularization weight
        input_weight = 0.05  # Lower weight for control input
        
        # Calculate direction to the target (for any target, not just static goals)
        target_dir = desired_state[0:2] - initial_state[0:2]
        target_dist = np.linalg.norm(target_dir)
        # Normalize direction if distance is significant
        if target_dist > 0.01:
            target_dir = target_dir / target_dist
        
        # Simulate system forward using the control sequence
        for i in range(Hp):
            # Determine which control input to use
            # After the control horizon, use the last control
            control_idx = min(i, Hc-1)
            u = u_sequence[control_idx]
            
            # Predict next state
            z[:, i+1] = self.robot_model.predict(z[:, i], u, dt)
            
            # Focus ONLY on position error (x, y)
            pos_error = desired_state[0:2] - z[0:2, i+1]
            
            # Quadratic position cost
            pos_cost = position_weight * float(np.dot(pos_error, pos_error))
            
            # For any trajectory, consider directional preference
            if i > 0:
                # Calculate movement direction
                move_dir = z[0:2, i+1] - z[0:2, i]
                move_dist = np.linalg.norm(move_dir)
                if move_dist > 0.01:
                    # Normalize movement direction
                    move_dir = move_dir / move_dist
                    # Calculate alignment with target direction
                    if target_dist > 0.01:  # Only if we have a meaningful target direction
                        alignment = np.dot(target_dir, move_dir)
                        # Add cost for poor alignment (higher when not aligned)
                        cost += position_weight * 0.5 * (1.0 - alignment)
            
            # Weight near-term predictions more heavily
            horizon_weight = 1.0 - 0.5 * (i / Hp)  # Weight decreases along the horizon
            cost += horizon_weight * pos_cost
            
            # Add control input regularization cost
            if i < Hc:
                control_cost = input_weight * float(np.dot(u, u))
                cost += control_cost
                
                # Add rate cost if not the first control
                if i > 0:
                    du = u - u_sequence[i-1]
                    rate_cost = 0.05 * float(np.dot(du, du))
                    cost += rate_cost
        
        # Add terminal state cost with higher weight
        pos_error_final = desired_state[0:2] - z[0:2, -1]
        final_pos_cost = terminal_weight * float(np.dot(pos_error_final, pos_error_final))
        cost += final_pos_cost
        
        # Store the predicted trajectory for visualization
        self.predicted_states = [z[:, i] for i in range(Hp+1)]
        
        return float(cost)
        
    def compute_sequence_gradient(self, u_sequence: np.ndarray, fx: float, 
                                 desired_state: np.ndarray, initial_state: np.ndarray, 
                                 dt: float, Hp: int, Hc: int) -> np.ndarray:
        """Compute gradient of cost function with respect to control sequence.
        
        Args:
            u_sequence: Sequence of control inputs (Hc x nu)
            fx: Current cost value
            desired_state: Desired state
            initial_state: Initial state
            dt: Time step
            Hp: Prediction horizon
            Hc: Control horizon
            
        Returns:
            Gradient of cost function with respect to control sequence
        """
        # Initialize gradient for the entire sequence
        gradient = np.zeros_like(u_sequence, dtype=float)
        base_eps = 1e-4
        
        # Compute gradient for each control input in the sequence
        for i in range(Hc):
            for j in range(len(u_sequence[i])):
                # Adaptive step size based on control input magnitude
                eps = base_eps * (1.0 + 0.1 * np.abs(u_sequence[i, j]))
                
                # Create perturbed sequences
                u_pos = u_sequence.copy()
                u_pos[i, j] += eps
                
                u_neg = u_sequence.copy()
                u_neg[i, j] -= eps
                
                # Clip to respect constraints
                u_pos[i] = np.clip(u_pos[i], self.params.u_min, self.params.u_max)
                u_neg[i] = np.clip(u_neg[i], self.params.u_min, self.params.u_max)
                
                # Compute costs with perturbed sequences
                fx_pos = self.compute_sequence_cost(desired_state, initial_state, u_pos, dt, Hp, Hc)
                fx_neg = self.compute_sequence_cost(desired_state, initial_state, u_neg, dt, Hp, Hc)
                
                # Central difference for more accurate gradient
                gradient[i, j] = float((fx_pos - fx_neg) / (2 * eps))
        
        return gradient
    
    def compute_cost(self, desired_state: np.ndarray, state: np.ndarray, 
                     u: np.ndarray, dt: float, Hp: int) -> float:
        """Compute cost function for MPC optimization.
        This is a legacy method kept for compatibility with other code that may call it.
        
        Args:
            desired_state: Desired state trajectory
            state: Current state
            u: Control input
            dt: Time step
            Hp: Prediction horizon
            
        Returns:
            Cost value
        """
        # Create a single-step control sequence for compatibility
        u_sequence = np.array([u])
        # Use the newer sequence-based cost function with control horizon = 1
        return self.compute_sequence_cost(desired_state, state, u_sequence, dt, Hp, 1)
    
    def compute_stability_cost(self, state: np.ndarray) -> float:
        """Compute stability cost based on vehicle dynamics.
        
        Args:
            state: Current state
            
        Returns:
            Stability cost
        """
        # Extract omega (yaw rate) from state
        omega = float(state[5])
        
        # Calculate slip angle beta (approximation)
        if np.abs(state[3]) > 0.001:  # Avoid division by zero
            beta = float(np.arctan2(state[4], state[3]))
        else:
            beta = 0.0
        
        # Calculate stability cost - reduced impact for better position tracking
        cost = float(beta**2 + 0.5 * omega**2)
        
        # Define stability bounds - relaxed for better position tracking
        beta_bound = 0.15  # Increased from 0.1
        beta_m = 0.25     # Increased from 0.2
        yaw_rate_bound = 1.0  # Increased from 0.8
        
        # Check if state is outside stability region
        if ((beta > beta_m * omega + beta_bound) or 
            (beta < beta_m * omega - beta_bound) or 
            (omega > yaw_rate_bound) or 
            (omega < -yaw_rate_bound)):
            return float(cost)
        else:
            return 0.0
    
    def compute_gradient(self, u: np.ndarray, fx: float, 
                          desired_state: np.ndarray, state: np.ndarray, 
                          dt: float, Hp: int) -> np.ndarray:
        """Compute gradient of cost function with respect to control inputs.
        This is a legacy method kept for compatibility with other code that may call it.
        
        Args:
            u: Control input
            fx: Current cost value
            desired_state: Desired state
            state: Current state
            dt: Time step
            Hp: Prediction horizon
            
        Returns:
            Gradient of cost function
        """
        # Create a single-step control sequence and compute gradient
        u_sequence = np.array([u])
        # Compute sequence gradient with control horizon = 1
        sequence_gradient = self.compute_sequence_gradient(u_sequence, fx, desired_state, state, dt, Hp, 1)
        # Return the first (and only) element of the sequence gradient
        return sequence_gradient[0]
