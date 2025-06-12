"""Bi-Level Model Predictive Control with Safety Constraints.

This implementation follows the approach in BiLevelUturn.m directly,
using a gradient-based optimization approach instead of CasADi.
"""

import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any
from src.models.robot import Robot4WSD
from src.safety.barrier import DistanceBarrier, YieldingBarrier, SpeedBarrier, AccelBarrier

logger = logging.getLogger(__name__)


@dataclass
class MPCParams:
    """MPC parameters from config"""
    Hp: int          # Prediction horizon
    dt: float        # Time step
    Q1: np.ndarray   # Position tracking weights
    Q2: np.ndarray   # Velocity tracking weights
    R: np.ndarray    # Control input weights
    u_min: np.ndarray  # Input lower bounds
    u_max: np.ndarray  # Input upper bounds
    du_min: np.ndarray  # Input rate lower bounds
    du_max: np.ndarray  # Input rate upper bounds

class BiLevelMPC:
    """Bi-Level Model Predictive Control implementation.
    
    The controller solves a bi-level optimization problem:
    - Upper level: Minimize tracking error and control effort
    - Lower level: Maximize safety margin through CBF constraints
    """
    
    def __init__(self, config: dict):
        """Initialize MPC controller.
        
        Args:
            config: Configuration
        """
        # Load configuration
        self.config = config

        # Set up MPC parameters - use MPC specific time_step from controller/mpc config
        self.params = MPCParams(
            Hp=self.config['controller']['mpc']['prediction_horizon'],
            dt=self.config['controller']['mpc']['time_step'],  # This is already correct
            Q1=np.array(self.config['controller']['mpc']['weights']['Q1']),
            Q2=np.array(self.config['controller']['mpc']['weights']['Q2']),
            R=np.array(self.config['controller']['mpc']['weights']['R']),
            u_min=np.concatenate([
                self.config['controller']['mpc']['constraints']['steering']['min'],
                self.config['controller']['mpc']['constraints']['voltage']['min']
            ]),
            u_max=np.concatenate([
                self.config['controller']['mpc']['constraints']['steering']['max'],
                self.config['controller']['mpc']['constraints']['voltage']['max']
            ]),
            du_min=np.concatenate([
                self.config['controller']['mpc']['constraints']['steering']['rate_min'],
                self.config['controller']['mpc']['constraints']['voltage']['rate_min']
            ]),
            du_max=np.concatenate([
                self.config['controller']['mpc']['constraints']['steering']['rate_max'],
                self.config['controller']['mpc']['constraints']['voltage']['rate_max']
            ])
        )
        # Initialize concrete barrier functions
        self.safety_barriers = [
            DistanceBarrier(config),
            YieldingBarrier(config),
            SpeedBarrier(config),
            AccelBarrier(config)
        ]
        
        # Initialize robot model for prediction
        self.robot_model = Robot4WSD(config)
        
        # Cache for last control input (for rate limiting)
        self.last_u = np.zeros(len(self.params.u_min))
        
        # Cache for warm starting the optimization
        self.last_solution = None
        
        # Momentum and adaptive learning rate parameters
        self.momentum = 0.9
        self.momentum_velocity = None
        self.adaptive_lr_patience = 3
        self.adaptive_lr_factor = 0.5
        self.adaptive_lr_min = 0.01
        self.cost_history = []
        self.patience_counter = 0
        
    def action(self, state: np.ndarray, desired_state: np.ndarray, safety_data: Dict[str, Any]) -> np.ndarray:
        """Compute control action using MPC.
        
        Args:
            state: Current robot state [x, y, θ, vx, vy, omega]
            desired_state: Desired robot state [x, y, θ, vx, vy, omega]
            safety_data: Safety-related data for the current simulation step
 
        Returns:
            Control action [δ_front, δ_rear, V_FL, V_FR, V_RL, V_RR]
            where δ are steering angles in radians and V are motor voltages.
        """
        try:
            # Ensure state arrays have consistent sizes
            if len(state.shape) != 1 or state.shape[0] < 6:
                logger.debug(f"Warning: Invalid state shape {state.shape}, padding with zeros")
                # Pad with zeros if needed
                full_state = np.zeros(6)
                full_state[:min(6, state.shape[0])] = state[:min(6, state.shape[0])]
                state = full_state
                
            # Ensure desired_state has correct dimensions
            if len(desired_state.shape) != 1:
                logger.debug(f"Warning: Invalid desired_state shape {desired_state.shape}, flattening")
                desired_state = desired_state.flatten()
                
            if desired_state.shape[0] < 6:
                logger.debug(f"Warning: Desired state too short {desired_state.shape}, padding")
                # Pad with zeros if needed
                full_desired_state = np.zeros(6)
                full_desired_state[:desired_state.shape[0]] = desired_state
                desired_state = full_desired_state
                
            # Solve MPC optimization
            u_opt = self.solve_mpc(state, desired_state)
            self.last_u = u_opt.copy()
            return u_opt
            
        except Exception as e:
            logger.debug(f"Error in MPC controller: {e}")
            # Fallback to last known good control or zero
            return self.last_u.copy() if self.last_u is not None else np.zeros(6)
        
    def solve_mpc(self, state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        """Solve bi-level MPC optimization using gradient-based approach.
        
        Args:
            state: Current robot state [x, y, θ, vx, vy, omega]
            desired_state: Target state to track
            
        Returns:
            Optimal control action [delta_front, delta_rear, V_FL, V_FR, V_RL, V_RR]
        """
        # Extract MPC parameters
        Hp = self.params.Hp
        dt = self.params.dt
        Q1 = self.params.Q1
        Q2 = self.params.Q2
        R = self.params.R
        u_min = self.params.u_min
        u_max = self.params.u_max
        du_min = self.params.du_min
        du_max = self.params.du_max
        n_u = len(u_min)
        
        # Get optimization settings from config
        max_iterations = int(self.config.get('optimization', {}).get('max_iterations', 20))
        learning_rate = float(self.config.get('optimization', {}).get('learning_rate', 1.0))
        min_accuracy = float(self.config.get('optimization', {}).get('min_accuracy', 1e-2))
        max_time = float(self.config.get('optimization', {}).get('max_time', 0.02))
        
        # Initial control guess - use last solution if available
        if self.last_solution is not None:
            u_current = self.last_solution[:, 0].copy()
        else:
            # Start with a reasonable initial control
            # Front steering to point toward goal
            dx = desired_state[0] - state[0]
            dy = desired_state[1] - state[1]
            target_angle = np.arctan2(dy, dx)
            initial_steer = 0.3 * np.sin(target_angle - state[2])
            
            # Voltage proportional to distance
            distance = np.sqrt(dx**2 + dy**2)
            initial_voltage = min(3.0, 0.3 * distance)
            
            u_current = np.array([
                initial_steer,           # Front steering
                0.0,                     # Rear steering
                initial_voltage,         # FL motor
                initial_voltage,         # FR motor
                initial_voltage,         # RL motor
                initial_voltage          # RR motor
            ])
        
        # Safety parameters
        rho0 = float(self.config.get('safety', {}).get('rho_0', 1.0))
        rho1 = float(self.config.get('safety', {}).get('rho_1', 2.0))
        theta0 = float(self.config.get('safety', {}).get('theta_0', np.pi/2))
        vmax = float(self.config.get('safety', {}).get('limits', {}).get('velocity', {}).get('max', 1.0))
        
        # Prepare human state data
        human_states = []
        
        # Add goal as a positional target for path planning with low weight
        if desired_state.shape[0] >= 2:
            human_states.append({
                'x': float(desired_state[0]),
                'y': float(desired_state[1]),
                'vx': float(desired_state[3]) if desired_state.shape[0] > 3 else 0.0,
                'vy': float(desired_state[4]) if desired_state.shape[0] > 4 else 0.0,
                'is_goal': True  # Flag to identify this as the goal, not an obstacle
            })
        
        # Add human obstacles from the configuration
        scenario_name = self.config['scenario']['name']
        scenario_config = self.config['scenario'].get(scenario_name, {})
        if 'humans' in scenario_config:
            human_positions = scenario_config['humans']['positions']
            human_velocities = scenario_config['humans']['velocities']
            
            # Loop through all defined human obstacles
            for i, (pos, vel) in enumerate(zip(human_positions, human_velocities)):
                # debug output to verify obstacle loading
                logger.debug(f"Loading human obstacle {i}: pos={pos}, vel={vel}")
                
                human_states.append({
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'vx': float(vel[0]),
                    'vy': float(vel[1]),
                    'is_goal': False  # This is an obstacle, not a goal
                })
        
        # Start optimization timer
        start_time = time.time()
        
        # Main optimization loop - bi-level approach
        iteration = 0
        gradient_norm = float('inf')
        best_cost = float('inf')
        best_u = u_current.copy()
        
        # For conjugate gradient descent
        last_gradient = np.zeros(n_u)
        conjugate_direction = np.zeros(n_u)
        beta = 0.5  # Conjugate gradient parameter
        
        # Initialize momentum velocity
        if self.momentum_velocity is None:
            self.momentum_velocity = np.zeros(n_u)
        
        while iteration < max_iterations and gradient_norm > min_accuracy and time.time() - start_time < max_time:
            # First level: Calculate cost for current control
            current_cost = self.upper_level_cost(state, desired_state, u_current, dt, Hp, Q1, Q2, R, human_states)
            
            # Compute gradient using finite differences
            gradient = self.calculate_gradient(state, desired_state, u_current, dt, Hp, Q1, Q2, R, human_states)
            gradient_norm = np.linalg.norm(gradient)
            
            # Update conjugate direction
            if iteration == 0:
                conjugate_direction = -gradient
            else:
                # Fletcher-Reeves formula for beta
                beta = np.dot(gradient, gradient) / max(np.dot(last_gradient, last_gradient), 1e-10)
                conjugate_direction = -gradient + beta * conjugate_direction
            
            # Store current gradient for next iteration
            last_gradient = gradient.copy()
            
            # Apply momentum
            self.momentum_velocity = self.momentum * self.momentum_velocity + (1 - self.momentum) * gradient
            momentum_adjusted_gradient = self.momentum_velocity
            
            # Line search with dynamic step size
            step_size = learning_rate
            u_new = u_current.copy()
            
            # Try different step sizes
            search_success = False
            for _ in range(5):  # Try a few different step sizes
                # Update using conjugate gradient direction
                u_trial = u_current + step_size * conjugate_direction
                
                # Apply constraints
                u_trial = np.clip(u_trial, u_min, u_max)
                
                # Apply rate constraints if not first control
                if self.last_u is not None:
                    rate_diff = u_trial - self.last_u
                    rate_diff = np.clip(rate_diff, du_min, du_max)
                    u_trial = self.last_u + rate_diff
                
                # Second level: Enforce safety constraints
                u_trial_safe = self.enforce_safety_constraints(state, u_trial, human_states, dt, rho0, rho1, theta0, vmax)
                
                # Evaluate cost with new control
                trial_cost = self.upper_level_cost(state, desired_state, u_trial_safe, dt, Hp, Q1, Q2, R, human_states)
                
                # Accept if better
                if trial_cost < current_cost:
                    u_new = u_trial_safe
                    current_cost = trial_cost
                    search_success = True
                    break
                else:
                    # Reduce step size if not improving
                    step_size *= 0.5
            
            # If line search failed to improve cost, add some exploration noise
            if not search_success and iteration > 0:
                logger.debug("Adding exploration noise...")
                noise = np.random.normal(0, 0.05, size=n_u)
                u_trial = u_current + noise
                u_trial = np.clip(u_trial, u_min, u_max)
                
                # Apply rate constraints
                if self.last_u is not None:
                    rate_diff = u_trial - self.last_u
                    rate_diff = np.clip(rate_diff, du_min, du_max)
                    u_trial = self.last_u + rate_diff
                
                # Enforce safety
                u_trial_safe = self.enforce_safety_constraints(state, u_trial, human_states, dt, rho0, rho1, theta0, vmax)
                
                # Evaluate cost
                trial_cost = self.upper_level_cost(state, desired_state, u_trial_safe, dt, Hp, Q1, Q2, R, human_states)
                
                # Accept if better
                if trial_cost < current_cost:
                    u_new = u_trial_safe
                    current_cost = trial_cost
            
            # Update best solution if improved
            if current_cost < best_cost:
                best_cost = current_cost
                best_u = u_new.copy()
                self.patience_counter = 0  # Reset patience counter
            else:
                self.patience_counter += 1
            
            # Adaptive learning rate adjustment
            if self.patience_counter >= self.adaptive_lr_patience:
                learning_rate = max(self.adaptive_lr_min, learning_rate * self.adaptive_lr_factor)
                logger.debug(f"Adaptive learning rate adjusted to {learning_rate:.4f}")
                self.patience_counter = 0
            
            # Update control for next iteration
            u_current = u_new
            
            # Increment counter
            iteration += 1
        
        # Record optimization stats
        optimization_time = time.time() - start_time
        logger.debug(f"BiLevelMPC: {iteration} iterations in {optimization_time*1000:.1f}ms, cost={best_cost:.4f}, |∇|={gradient_norm:.4f}")
        
        # Update last control and solution
        self.last_u = best_u.copy()
        
        # For warm starting, create a prediction over the horizon
        if self.last_solution is None:
            self.last_solution = np.zeros((n_u, Hp))
        
        # Shift solution and append last control
        self.last_solution = np.column_stack([self.last_solution[:, 1:], best_u.reshape(-1, 1)])
        
        return best_u
        
    def upper_level_cost(self, state: np.ndarray, desired_state: np.ndarray, 
                       control: np.ndarray, dt: float, horizon: int,
                       Q1: np.ndarray, Q2: np.ndarray, R: np.ndarray, human_states: list = None) -> float:
        """Calculate upper-level cost function (trajectory tracking and control effort).
        
        Args:
            state: Current state [x, y, θ, vx, vy, omega]
            desired_state: Target state
            control: Control input [delta_front, delta_rear, V_FL, V_FR, V_RL, V_RR]
            dt: Time step
            horizon: Prediction horizon
            Q1, Q2, R: Weight matrices
            human_states: List of human state dictionaries
            
        Returns:
            Cost value
        """
        # Initialize human_states if not provided
        if human_states is None:
            human_states = []
        # Initialize cost and predicted state
        total_cost = 0.0
        predicted_state = state.copy()
        
        # Direction from current to goal 
        dx = desired_state[0] - state[0]
        dy = desired_state[1] - state[1]
        distance_to_goal = np.sqrt(dx**2 + dy**2)
        desired_heading = np.arctan2(dy, dx)
        
        # Angle error weight (increases as we get closer to goal)
        angle_weight = 1.0 + 3.0 * np.exp(-0.2 * max(0, distance_to_goal - 1.0))
        
        # Simulate forward over the horizon
        for i in range(horizon):
            # Position error cost with modified weights for angle
            pos_error = predicted_state[0:3] - desired_state[0:3]
            
            # Increase weight on heading as we get closer
            weighted_Q1 = Q1.copy()
            weighted_Q1[2] *= angle_weight  # Increase heading error weight
            
            pos_cost = float(np.sum(weighted_Q1 * pos_error**2))
            
            # Add specific heading error cost
            current_heading = predicted_state[2]
            heading_error = np.arctan2(np.sin(desired_heading - current_heading), 
                                     np.cos(desired_heading - current_heading))
            heading_cost = 2.0 * angle_weight * heading_error**2
            
            total_cost += pos_cost + heading_cost
            
            # Velocity error cost
            vel_error = predicted_state[3:6] - desired_state[3:6]
            vel_cost = float(np.sum(Q2 * vel_error**2))
            total_cost += vel_cost
            
            # Control input cost with higher weight on steering
            steering_cost = float(10.0 * np.sum(control[0:2]**2))
            voltage_cost = float(np.sum(R[2:] * control[2:]**2))
            total_cost += steering_cost + voltage_cost
            
            # Special cost to discourage zero control when far from goal
            if distance_to_goal > 0.5 and np.sum(np.abs(control[2:6])) < 0.1:
                total_cost += 100.0 * np.exp(-0.1 * np.sum(np.abs(control[2:6])))
            
            # Predict next state
            predicted_state = self.robot_model.predict(predicted_state, control, dt)
        
            # Update distance and heading for next iteration
            dx = desired_state[0] - predicted_state[0]
            dy = desired_state[1] - predicted_state[1]
            distance_to_goal = np.sqrt(dx**2 + dy**2)
            desired_heading = np.arctan2(dy, dx)
            
            # Update angle weight
            angle_weight = 1.0 + 3.0 * np.exp(-0.2 * max(0, distance_to_goal - 1.0))
            
        # Terminal cost with more aggressive weighting
        end_dx = desired_state[0] - predicted_state[0]
        end_dy = desired_state[1] - predicted_state[1]
        end_distance = np.sqrt(end_dx**2 + end_dy**2)
        
        # Use a more aggressive cost function that increases as we get closer
        # to ensure the robot reaches the goal precisely
        terminal_cost = 20.0 * end_distance  # Linear term for far away
        if end_distance < 1.0:  # When within 1m, use quadratic to ensure convergence
            terminal_cost += 30.0 * (end_distance ** 2)
        
        # Add orientation cost that becomes more important as we approach the goal
        desired_heading = np.arctan2(end_dy, end_dx)
        heading_error = np.arctan2(np.sin(desired_heading - predicted_state[2]),
                                 np.cos(desired_heading - predicted_state[2]))
        
        # Scale orientation cost based on distance (more important when close)
        orientation_weight = 5.0 * np.exp(-2.0 * end_distance)
        terminal_cost += orientation_weight * (heading_error ** 2)
        
        # Velocity penalty to ensure smooth stopping
        current_speed = np.sqrt(predicted_state[3]**2 + predicted_state[4]**2)
        if end_distance < 2.0:  # Start slowing down when within 2m
            # Quadratic penalty for not matching desired final velocity (0)
            terminal_cost += 5.0 * current_speed ** 2
            terminal_cost += 1.0 * (predicted_state[3]**2 + predicted_state[4]**2)
            
        total_cost += terminal_cost
        
        # Penalize steering angle constraint violations
        max_steering = self.params.u_max[0]
        if abs(control[0]) > max_steering:
            total_cost += 10.0 * (abs(control[0]) - max_steering)
        
        # Add stability cost term
        # Calculate sideslip angle and yaw rate
        vx, vy = predicted_state[3], predicted_state[4]
        speed = np.sqrt(vx**2 + vy**2)
        omega = predicted_state[5]
        
        if speed > 0.01:  # Only calculate sideslip if moving
            beta = np.abs(np.arctan2(vy, vx))
            # Penalty for high sideslip angles
            beta_bound = 0.1
            beta_m = 0.2
            if beta > (beta_m * abs(omega) + beta_bound):
                total_cost += 5.0 * (beta - (beta_m * abs(omega) + beta_bound))**2
        
        # Add human proximity cost
        total_cost += self.calculate_human_proximity_cost(predicted_state[0], predicted_state[1], human_states)
        
        return float(total_cost)
    
    def calculate_gradient(self, state: np.ndarray, desired_state: np.ndarray, 
                         control: np.ndarray, dt: float, horizon: int, 
                         Q1: np.ndarray, Q2: np.ndarray, R: np.ndarray, human_states: list = None) -> np.ndarray:
        """Calculate gradient of the cost function using finite differences.
        
        Args:
            state: Current state
            desired_state: Target state
            control: Current control
            dt: Time step
            horizon: Prediction horizon
            Q1, Q2, R: Weight matrices
            human_states: List of human state dictionaries
            
        Returns:
            Gradient of cost with respect to control
        """
        # Get the human states from the current solve context
        human_states = []
        
        # Add goal as a positional target
        if desired_state.shape[0] >= 2:
            human_states.append({
                'x': float(desired_state[0]),
                'y': float(desired_state[1]),
                'vx': float(desired_state[3]) if desired_state.shape[0] > 3 else 0.0,
                'vy': float(desired_state[4]) if desired_state.shape[0] > 4 else 0.0,
                'is_goal': True  # Flag to identify this as the goal, not an obstacle
            })
        
        # Add human obstacles from the configuration
        if self.config['scenario']['name'] == 'to_goal' and 'humans' in self.config['scenario']['to_goal']:
            human_positions = self.config['scenario']['to_goal']['humans']['positions']
            human_velocities = self.config['scenario']['to_goal']['humans']['velocities']
            
            # Loop through all defined human obstacles
            for i, (pos, vel) in enumerate(zip(human_positions, human_velocities)):
                human_states.append({
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'vx': float(vel[0]),
                    'vy': float(vel[1]),
                    'is_goal': False  # This is an obstacle, not a goal
                })
        
        # Calculate baseline cost
        base_cost = self.upper_level_cost(state, desired_state, control, dt, horizon, Q1, Q2, R, human_states)
        
        # Define perturbation size
        epsilon = 1e-3  # Increased for better numerical gradients
        
        # Initialize gradient vector
        n_u = len(control)
        gradient = np.zeros(n_u)
        
        # Calculate gradient for each control input using central difference (more accurate)
        for i in range(n_u):
            # Create forward perturbed control
            perturbed_forward = control.copy()
            perturbed_forward[i] += epsilon
            
            # Create backward perturbed control
            perturbed_backward = control.copy()
            perturbed_backward[i] -= epsilon
            
            # Calculate costs for both perturbations
            forward_cost = self.upper_level_cost(state, desired_state, perturbed_forward, dt, horizon, Q1, Q2, R, human_states)
            backward_cost = self.upper_level_cost(state, desired_state, perturbed_backward, dt, horizon, Q1, Q2, R, human_states)
            
            # Calculate gradient component using central difference
            gradient[i] = (forward_cost - backward_cost) / (2 * epsilon)
        
        # Print gradient information for debugging
        logger.debug(f"Gradient: mag={np.linalg.norm(gradient):.4f}, values={np.round(gradient, 4)}")
        
        return gradient
    
    def enforce_safety_constraints(self, state: np.ndarray, control: np.ndarray, 
                                 human_states: list, dt: float, 
                                 rho0: float, rho1: float, theta0: float, vmax: float) -> np.ndarray:
        """Enforce safety constraints by modifying the control input.
        
        Args:
            state: Current state
            control: Current control input
            human_states: List of human state dictionaries
            dt: Time step
            rho0, rho1, theta0: Safety distance parameters
            vmax: Maximum allowed velocity
            
        Returns:
            Modified control input that satisfies safety constraints
        """
        safe_control = control.copy()
        
        # Check if there are humans to avoid
        if not human_states:
            return safe_control
        
        # Extract state components
        x, y, theta = state[0:3]
        vx, vy, omega = state[3:6]
        
        # Current speed
        current_speed = np.sqrt(vx**2 + vy**2)
        
        # 1. Speed constraint - from the BiLevelUturn.m reference
        if current_speed > vmax:
            # Scale down motor voltages proportionally to reduce speed
            speed_ratio = vmax / max(current_speed, 0.01)
            safe_control[2:6] = safe_control[2:6] * speed_ratio**2
        
        # 2. First pass: find the closest human obstacle (excluding the goal)
        closest_human = None
        min_distance = float('inf')
        
        # 3. Implement improved handling of humans vs goal
        for human in human_states:
            # Calculate distance to human
            dx = human['x'] - x
            dy = human['y'] - y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Skip the goal point in obstacle avoidance - it's a target, not an obstacle
            is_goal = human.get('is_goal', False)
            
            if not is_goal and distance < min_distance:
                min_distance = distance
                closest_human = human
        
        # 4. Second pass: handle obstacle avoidance for the closest human
        if closest_human is not None:
            dx = closest_human['x'] - x
            dy = closest_human['y'] - y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate angle to human relative to robot heading
            rel_angle = np.arctan2(dy, dx) - theta
            rel_angle = np.arctan2(np.sin(rel_angle), np.cos(rel_angle))  # Normalize to [-π, π]
            
            # Calculate relative velocity
            rel_vx = closest_human['vx'] - vx
            rel_vy = closest_human['vy'] - vy
            rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
            
            # Adjust min distance based on relative velocity (closing speed)
            closing_velocity = max(0, -(rel_vx * dx + rel_vy * dy) / max(distance, 0.01))
            dynamic_safety_distance = rho0 + 0.5 * closing_velocity  # Add safety margin based on closing speed
            
            # Determine safety threshold based on angle
            min_safe_distance = dynamic_safety_distance if abs(rel_angle) < theta0 else rho1
            
            # Implement avoidance action when too close to obstacle
            if distance < min_safe_distance * 1.2:  # Additional margin
                logger.debug(f"Safety constraint active: distance={distance:.2f}m, human at ({closest_human['x']:.1f}, {closest_human['y']:.1f})")
                
                # Calculate steering adjustment to avoid obstacle
                # If obstacle is in front, steer away from it
                if abs(rel_angle) < theta0:  
                    # Direction to steer (away from obstacle)
                    steer_direction = -np.sign(rel_angle)
                    
                    # Scale steering based on how close we are to the minimum distance
                    intensity = 1.0 - min(1.0, distance / min_safe_distance)
                    
                    # Apply stronger steering correction when closer
                    steering_correction = steer_direction * 0.3 * intensity
                    
                    # Apply to front steering
                    safe_control[0] = steer_direction * min(abs(safe_control[0] + steering_correction), 0.4)
                    
                # If in danger zone, also reduce speed based on proximity
                speed_scale = min(1.0, (distance / min_safe_distance) ** 2)
                safe_control[2:6] *= max(0.3, speed_scale)  # At least 30% of original speed
        
        # 5. Calculate sideslip angle for stability assessment
        beta = abs(np.arctan2(vy, vx)) if current_speed > 0.01 else 0.0
        r = abs(omega)  # Yaw rate magnitude
        
        # Stability bounds
        beta_bound = 0.1
        beta_m = 0.2
        yaw_rate_bound = 0.8
        
        # Check stability constraints
        if (beta > (beta_m * r + beta_bound)) or (r > yaw_rate_bound):
            # Reduce steering to improve stability
            safe_control[0] *= 0.7  # Reduce by 30%
            safe_control[1] *= 0.7
            
            # Also reduce speed if stability is compromised
            safe_control[2:6] *= 0.8
            
        return safe_control
    
    def calculate_human_proximity_cost(self, pos_x: float, pos_y: float, 
                                human_states: list, is_terminal: bool = False) -> float:
        """Calculate cost associated with proximity to human obstacles.
        
        Args:
            pos_x: X position of the robot
            pos_y: Y position of the robot
            human_states: List of human state dictionaries
            is_terminal: Whether this is for terminal cost (higher weight)
            
        Returns:
            Cost value penalizing proximity to obstacles
        """
        if not human_states:
            return 0.0
            
        # Minimum distance we want to maintain
        safe_distance = float(self.config.get('safety', {}).get('rho_0', 1.5))
        
        total_cost = 0.0
        
        for human in human_states:
            # Skip the goal point - we want to get closer to it, not away from it
            if human.get('is_goal', False):
                continue
                
            # Calculate distance to human
            dx = human['x'] - pos_x
            dy = human['y'] - pos_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Exponential cost that grows rapidly as distance decreases
            if distance < safe_distance * 2.0:  # Only consider obstacles within 2x safe distance
                # Cost function that grows exponentially as we get closer
                proximity_cost = 30.0 * np.exp(-2.0 * (distance / safe_distance))
                
                # Higher penalty for terminal state to encourage planning paths away from obstacles
                if is_terminal:
                    proximity_cost *= 2.0
                    
                total_cost += proximity_cost
        
        return total_cost
