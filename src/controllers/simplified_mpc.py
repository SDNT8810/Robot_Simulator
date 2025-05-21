"""Simplified Bi-Level Model Predictive Control without CBF constraints."""

import numpy as np
import casadi as ca
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

class SimplifiedMPC:
    """Simplified MPC controller without CBF constraints (for testing).
    """
    
    def __init__(self, config: dict):
        """Initialize MPC controller.
        
        Args:
            config: Configuration
        """
        # Load configuration
        self.config = config

        # Set up MPC parameters
        # Convert weight lists to diagonal matrices for proper matrix multiplication
        q1_list = self.config['controller']['mpc']['weights']['Q1']
        q2_list = self.config['controller']['mpc']['weights']['Q2']
        r_list = self.config['controller']['mpc']['weights']['R']
        
        self.params = MPCParams(
            Hp=self.config['controller']['mpc']['prediction_horizon'],
            dt=self.config['controller']['mpc']['time_step'],
            Q1=np.diag(q1_list),  # Make diagonal 3x3 matrix from list
            Q2=np.diag(q2_list),  # Make diagonal 3x3 matrix from list
            R=np.diag(r_list),    # Make diagonal matrix from list
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
        
        # Initialize robot model for prediction
        self.robot_model = Robot4WSD(config)
        
        # Cache for last control input (for rate limiting)
        self.last_u = np.zeros(len(self.params.u_min))
        
        # Cache for warm starting the optimization
        self.last_solution = None
        
        # Add integral action for tracking moving targets - only for position (x,y)
        self.integral_error = np.zeros(2, dtype=float)  # Position error integral
        self.integral_weight = np.array([0.2, 0.2], dtype=float)  # Weight for integral action
        self.integral_limit = 3.0  # Anti-windup limit
        
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
        
        # Update integral error for position ONLY (x,y)
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
        
        # Solve MPC problem with modified desired state
        u_opt = self.solve_mpc(state_float, modified_desired_state)
        
        # Save current control for next iteration (for rate limiting)
        self.last_u = u_opt.copy()
        
        return u_opt

    def solve_mpc(self, state: np.ndarray, desired_state: np.ndarray) -> np.ndarray:
        """Solve the Simplified MPC optimization problem.
        
        Args:
            state: Current state of the robot
            desired_state: Desired state to track
            
        Returns:
            Optimal control action
        """
        # Extract MPC parameters from config
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
        
        # Calculate position error and magnitude
        pos_error = desired_state[0:2] - state[0:2]
        error_magnitude = float(np.linalg.norm(pos_error))

        # Initial guess for control (warm start)
        if self.last_solution is not None:
            u_init = self.last_solution
        else:
            u_init = np.zeros((n_u, Hp))
            
        # If we're far from the goal and the initial control is zero, set a good initial guess
        if error_magnitude > 1.0 and np.all(np.abs(u_init) < 0.01):
            # Calculate desired heading
            heading = np.arctan2(pos_error[1], pos_error[0])
            
            # Create a reasonable initial guess - front wheel steers toward goal
            initial_steering = np.clip(heading, -0.5, 0.5)
            
            # Set initial voltage based on distance
            initial_voltage = min(8.0, float(error_magnitude))
            
            # Set steering for all prediction steps
            u_init[0, :] = initial_steering  # Front steering
            u_init[1, :] = 0.0  # Rear steering stays neutral
            
            # Set voltages for all prediction steps (if there are voltage controls)
            if u_init.shape[0] > 2:
                u_init[2:6, :] = initial_voltage

        # Set up optimization variables
        opti = ca.Opti()
        U = opti.variable(n_u, Hp)  # Control sequence
        X = opti.variable(6, Hp+1)  # State trajectory (x, y, theta, vx, vy, omega)

        # Initial state constraint
        opti.subject_to(X[:,0] == state)

        # Cost function
        cost = 0
        for k in range(Hp):
            # Tracking error for position (first 3 elements)
            pos_err = X[0:3, k] - desired_state[0:3]  # shape (3,)
            # Use quadratic cost for position error
            cost += ca.mtimes([pos_err.T, Q1, pos_err])
            
            # Add velocity error (elements 3-5) if dimensions match
            # Make sure we're comparing vectors of the same dimension
            if Q2.shape[0] == 3 and desired_state.shape[0] >= 6:
                vel_err = X[3:6, k] - desired_state[3:6]
                cost += ca.mtimes([vel_err.T, Q2, vel_err])
            
            # Add control input cost
            cost += ca.mtimes([U[:,k].T, R, U[:,k]])
            
            # Dynamics constraint
            x_next = self.robot_model.predict(X[:,k], U[:,k], dt)
            opti.subject_to(X[:,k+1] == x_next)
            
            # Input constraints
            opti.subject_to(u_min <= U[:,k])
            opti.subject_to(U[:,k] <= u_max)
            
            # Input rate constraints
            if k == 0:
                u_prev = self.last_u
            else:
                u_prev = U[:,k-1]
            opti.subject_to(du_min <= U[:,k] - u_prev)
            opti.subject_to(U[:,k] - u_prev <= du_max)
            
        # Terminal cost - ensure consistent dimensions
        pos_err = X[0:3, Hp] - desired_state[0:3]
        cost += ca.mtimes([pos_err.T, Q1, pos_err])
        
        # Add terminal velocity cost if dimensions match
        if Q2.shape[0] == 3 and desired_state.shape[0] >= 6:
            vel_err = X[3:6, Hp] - desired_state[3:6]
            cost += ca.mtimes([vel_err.T, Q2, vel_err])
        
        opti.minimize(cost)

        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
        opti.solver('ipopt', opts)
        try:
            sol = opti.solve()
            u_opt = np.array(sol.value(U[:,0])).flatten()
            self.last_solution = np.array(sol.value(U))
        except RuntimeError:
            # Infeasible: fallback to last or zero input
            u_opt = self.last_u if np.any(self.last_u) else np.zeros(n_u)
        return u_opt
