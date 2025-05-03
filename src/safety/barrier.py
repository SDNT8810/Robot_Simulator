"""Control Barrier Functions for robot safety."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Tuple
# from src.models.input import RobotInput
from src.utils.config import Load_Config

class ControlBarrierFunction(ABC):
    """Base class for Control Barrier Functions (CBFs).
    
    Each CBF h(x) defines a safety constraint h(x) ≥ 0. The time derivative
                      ḣ(x,u) must satisfy ḣ(x,u) + αh(x) ≥ 0 for some α > 0 to ensure safety.
    """
    
    def __init__(self, config: Load_Config):
        """Initialize CBF parameters from config."""
        self.alpha = config['safety']['alpha']  # CBF parameter
        self.uncertainty_bound = config['robot']['slip']['uncertainty']  # Uncertainty bound
    
        self.input_ = None  # Placeholder for robot input
        self.human_state = None  # Placeholder for human state
        self.config = config  # Store config for later use
        self.state = type('State', (object,), {})()
        self.state.x = 0.0
        self.state.y = 0.0
        self.state.theta = 0.0
        self.state.vx = 0.0
        self.state.vy = 0.0
        self.state.omega = 0.0
        self.state.i_fl = 0.0
        self.state.i_fr = 0.0
        self.state.i_rl = 0.0
        self.state.i_rr = 0.0 
        
    def get_adaptive_alpha(self) -> float:
        """Get adaptive CBF parameter based on robot state.
        
        The α parameter increases with robot speed to provide stronger
        safety guarantees at higher velocities.
        
        Args:
            state: Current robot state
            
        Returns:
            Adaptive α value
        """
        
        # self.x, self.y, self.theta = state_vector[0:3]
        # self.vx, self.vy, self.omega = state_vector[3:6]
        # self.i_fl, self.i_fr, self.i_rl, self.i_rr = state_vector[6:10]


        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        return self.alpha * (1.0 + 0.5 * v)  # Base value plus velocity-dependent term
        
    def get_uncertainty_margin(self) -> float:
        """Calculate safety margin to account for model uncertainty.
        
        Args:
            state: Current robot state
            
        Returns:
            Additional safety margin based on uncertainty
        """
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        return self.uncertainty_bound * v  # Uncertainty increases with speed
        
    def compute_robust_barrier(self, h_nominal: float, 
                             uncertainty_margin: float) -> float:
        """Compute robust barrier value accounting for uncertainty.
        
        Args:
            h_nominal: Nominal barrier function value
            uncertainty_margin: Safety margin for uncertainty
            
        Returns:
            Robust barrier value
        """
        return h_nominal - uncertainty_margin

    @abstractmethod
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate the CBF h(x).
        
        Args:
            human_state: Dictionary with human state info (position, velocity)
        
        Returns:
            Value of h(x). Positive values indicate safety.
        """
        pass
        
    @abstractmethod
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate the time derivative ḣ(x,u).
        
        Args:
            human_state: Dictionary with human state info
            
        Returns:
            Value of ḣ(x,u)
        """
        pass
        
    def verify_safety(self, human_state: Dict[str, float]) -> bool:
        """Check if the CBF condition is satisfied.
        
        Args:
            human_state: Dictionary with human state info
            
        Returns:
            True if ḣ(x,u) + αh(x) ≥ 0
        """
        h_val = self.h(human_state)
        uncertainty_margin = self.get_uncertainty_margin()
        h_robust = self.compute_robust_barrier(h_val, uncertainty_margin)
        
        h_dot_val = self.h_dot(human_state)
        alpha = self.get_adaptive_alpha()
        
        return h_dot_val + alpha * h_robust >= 0

class DistanceBarrier(ControlBarrierFunction):
    """Distance-based CBF for maintaining safe separation from humans."""
    
    def __init__(self, config: Load_Config):
        """Initialize distance CBF parameters."""
        super().__init__(config)
        # Base safety distances
        self.rho_0 = config['safety']['rho_0']
        self.rho_1 = config['safety']['rho_1']
        self.theta_0 = config['safety']['theta_0']
        
        # Dynamic safety margin parameters
        self.margin_slope = config['safety']['cbf_dynamics']['margin_slope']
        
    def _get_dynamic_safety_distance(self, base_distance: float) -> float:
        """Compute velocity-dependent safety distance.
        
        The safety distance increases with robot speed to account for
        longer stopping distances and reaction times at higher velocities.
        
        Args:
            base_distance: Base safety distance (rho_0 or rho_1)
            
        Returns:
            Adjusted safety distance
        """
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        dynamic_margin = self.margin_slope * v
        return base_distance + dynamic_margin
        
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate distance-based CBF with dynamic safety margins.
        
        h_dist = ρ - ρ_dyn if |θ| < θ_0 (in front)
               = ρ - ρ_dyn if |θ| ≥ θ_0 (to side)
        where ρ is distance to human, θ is relative angle,
        and ρ_dyn is the velocity-dependent safety distance
        """
        # Get relative position
        dx = human_state['x'] - self.state.x
        dy = human_state['y'] - self.state.y
        
        # Calculate distance and angle
        rho = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx) - self.state.theta
        
        # Choose threshold based on angle
        base_distance = self.rho_0 if abs(theta) < self.theta_0 else self.rho_1
        rho_min = self._get_dynamic_safety_distance(base_distance)
        
        return rho - rho_min
        
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of distance CBF.
        
                        ḣ_dist = d/dt(ρ) = (v_r · r_unit)
        where v_r is relative velocity, r_unit is unit vector from robot to human
        """
        # Get relative position and velocity
        dx = human_state['x'] - self.state.x
        dy = human_state['y'] - self.state.y
        dvx = human_state.get('vx', 0) - self.state.vx
        dvy = human_state.get('vy', 0) - self.state.vy
        
        # Calculate distance and unit vector
        rho = np.sqrt(dx**2 + dy**2)
        rx = dx / rho  # Unit vector x
        ry = dy / rho  # Unit vector y
        
        # Project relative velocity onto unit vector
        return dvx * rx + dvy * ry

class YieldingBarrier(ControlBarrierFunction):
    """Yielding behavior CBF for giving right of way to humans."""
    
    def __init__(self, config: Load_Config):
        """Initialize yielding CBF parameters."""
        super().__init__(config)
        # Load speed parameters
        self.nu_base = config['safety']['limits']['nu_max']['base']
        self.nu_slope = config['safety']['limits']['nu_max']['slope']
        self.nu_min = config['safety']['limits']['nu_max']['min']
        self.nu_max = config['safety']['limits']['nu_max']['max']
        
        
        # Load distance thresholds
        self.rho_0 = config['safety']['rho_0']
        self.rho_1 = config['safety']['rho_1']
        self.theta_0 = config['safety']['theta_0']
        
        # Load yielding zone parameters
        self.yield_start = config['safety']['yielding']['start_distance']
        self.yield_stop = config['safety']['yielding']['stop_distance']
        
        # Load smooth activation parameters
        self.activation_sharpness = config['safety']['cbf_dynamics']['activation_sharpness']
        
    def _smooth_activation(self, x: float, x0: float, width: float = 1.0) -> float:
        """Compute smooth activation function using sigmoid.
        
        Args:
            x: Input value
            x0: Activation threshold
            width: Transition width
            
        Returns:
            Smooth activation value in [0,1]
        """
        return 1.0 / (1.0 + np.exp(-self.activation_sharpness * (x - x0) / width))
        
    def _compute_max_speed(self, rho: float, v_approach: float) -> float:
        """Compute maximum allowed speed based on distance and approach velocity.
        
        Args:
            rho: Distance to human [m]
            v_approach: Approach velocity (negative when getting closer) [m/s]
            
        Returns:
            Maximum allowed speed [m/s]
        """
        # Base speed limit from linear interpolation
        progress = (rho - self.yield_stop) / (self.yield_start - self.yield_stop)
        progress = max(0.0, min(1.0, progress))
        v_max_base = self.nu_min + progress * (self.nu_base - self.nu_min)
        
        # Additional reduction based on approach velocity
        approach_factor = self._smooth_activation(-v_approach, 0.5, 1.0)
        v_max = v_max_base * approach_factor
        
        return max(self.nu_min, v_max)
        
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate yielding CBF with smooth activation.
        
        h_yield = ν_max(ρ,v_approach) - v
        where v is robot speed, ν_max is maximum allowed speed,
        which depends on both distance and approach velocity
        """
        # Calculate distance and approach velocity
        dx = human_state['x'] - self.state.x
        dy = human_state['y'] - self.state.y
        rho = np.sqrt(dx**2 + dy**2)
        
        dvx = human_state.get('vx', 0) - self.state.vx
        dvy = human_state.get('vy', 0) - self.state.vy
        v_approach = -(dx * dvx + dy * dvy) / rho  # Negative when getting closer
        
        # Calculate robot speed
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        
        # Compute maximum allowed speed with smooth activation
        v_max = self._compute_max_speed(rho, v_approach)
        
        # Add uncertainty margin for robustness
        uncertainty_margin = self.get_uncertainty_margin(self.state)
        v_max = max(self.nu_min, v_max - uncertainty_margin)
        
        return v_max - v

    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of yielding CBF.
        
                    ḣ_yield = d/dt(ν_max(ρ)) - d/dt(v)
        """
        # Calculate distance and its rate of change
        dx = human_state['x'] - self.state.x
        dy = human_state['y'] - self.state.y
        rho = np.sqrt(dx**2 + dy**2)
        
        dvx = human_state.get('vx', 0) - self.state.vx
        dvy = human_state.get('vy', 0) - self.state.vy
        rho_dot = (dx * dvx + dy * dvy) / rho
        
        # Calculate speed and acceleration
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        a = (self.state.vx * dvx + self.state.vy * dvy) / v if v > 0 else 0
        
        # Return CBF derivative
        return self.nu_slope * rho_dot - a

class SpeedBarrier(ControlBarrierFunction):
    """Speed limit CBF for maintaining safe velocities."""
    
    def __init__(self, config: Load_Config):
        """Initialize speed CBF parameters."""
        super().__init__(config)
        # Base speed limits from safety parameters
        self.nu_max = config['safety']['limits']['nu_max']['max']
        self.nu_min = config['safety']['limits']['nu_max']['min']
        self.nu_base = config['safety']['limits']['nu_max']['base']
        self.nu_slope = config['safety']['limits']['nu_max']['slope']
        
        # Dynamic speed limit parameters
        self.curvature_factor = config['safety']['cbf_dynamics']['curvature_factor']
        self.angular_velocity_limit = config['safety']['cbf_dynamics']['angular_velocity_limit']
        
    def _get_dynamic_speed_limit(self) -> float:
        """Compute dynamic speed limit based on robot state."""
        # Start with base speed limit
        v_max = min(self.nu_max, self.nu_base)
        
        # Smoother speed reduction based on angular velocity
        omega_normalized = abs(self.state.omega) / self.angular_velocity_limit
        omega_factor = max(0.3, 1.0 - 0.3 * omega_normalized)  # Less aggressive reduction
        
        # Apply curvature factor with smoother transition
        v_max = v_max * (self.curvature_factor + (1.0 - self.curvature_factor) * omega_factor)
        
        # Minimum guaranteed speed
        v_max = max(0.1, v_max)  # Always allow some minimal motion
        
        return v_max
    
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate speed CBF with dynamic limits.
        
        h_speed = v_max_dyn^2 - v^2
        Barrier activates (h < 0) when speed exceeds limit
        """
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        v_max = self._get_dynamic_speed_limit()
        
        # CBF value becomes negative when speed exceeds limit
        # Changed order to make barrier more permissive
        return v_max**2 - v**2
    
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of speed CBF."""
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        if v < 1e-6:  # Avoid division by zero
            return 0.0
            
        # Compute acceleration from motor inputs
        ax = self.state.get_motor_current(0) * self.state.torque_constant / self.state.mass
        ay = self.state.get_motor_current(1) * self.state.torque_constant / self.state.mass
        
        # Account for dynamic speed limit changes
        v_max = self._get_dynamic_speed_limit()
        v_max_dot = -0.3 * v_max * self.state.omega / self.angular_velocity_limit  # Reduced sensitivity
        
        # Project accelerations onto velocity direction
        v_dot = (self.state.vx * ax + self.state.vy * ay) / v
        
        # Changed order to match h(x) definition
        return 2 * v_max * v_max_dot - 2 * v * v_dot

class AccelBarrier(ControlBarrierFunction):
    """Acceleration limit CBF for smooth motion."""
    
    def __init__(self, config: Load_Config):
        """Initialize acceleration CBF parameters."""
        super().__init__(config)
        self.a_base = config['safety']['limits']['acceleration']['base']
        self.a_slope = config['safety']['limits']['acceleration']['slope']
        self.a_min = config['safety']['limits']['acceleration']['min']
        
        # Dynamic acceleration parameters
        self.velocity_factor = config['safety']['cbf_dynamics']['velocity_factor']
        self.slip_threshold = config['robot']['slip']['lambda_max']
        
    def _compute_max_accel(self, rho: float) -> float:
        """Compute maximum allowed acceleration based on distance and state.
        
        The acceleration limit adapts based on:
        1. Distance to human (base behavior)
        2. Current velocity (reduced at higher speeds)
        3. Estimated slip conditions
        
        Args:
            rho: Distance to human [m]
            
        Returns:
            Maximum allowed acceleration [m/s²]
        """
        # Base acceleration from distance with less reduction
        a_max_base = max(self.a_base + self.a_slope * rho, self.a_min)
        
        # Gentler velocity-based reduction
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        velocity_reduction = 1.0 / (1.0 + 0.5 * self.velocity_factor * v)
        
        # More permissive slip handling
        slip_ratio = min(abs(self.state.vx / (self.state.omega * self.state.wheel_radius + 1e-6)), 
                        self.slip_threshold)
        slip_factor = max(0.5, 1.0 - (slip_ratio / self.slip_threshold))
        
        # Reduced uncertainty margin for more permissive acceleration
        uncertainty_margin = 0.5 * self.get_uncertainty_margin(self.state)
        a_max = a_max_base * velocity_reduction * slip_factor
        return max(self.a_min, a_max - uncertainty_margin)
        
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate acceleration CBF with dynamic limits.
        
        h_accel = a^2 - a_max(ρ,v,λ)^2
        where a_max depends on distance ρ, velocity v, and slip ratio λ
        """
        # Calculate distance to human
        dx = human_state['x'] - self.state.x
        dy = human_state['y'] - self.state.y
        rho = np.sqrt(dx**2 + dy**2)
        
        # Get current acceleration
        ax = self.state.get_motor_current(0) * self.state.torque_constant / self.state.mass
        ay = self.state.get_motor_current(1) * self.state.torque_constant / self.state.mass
        a = np.sqrt(ax**2 + ay**2)
        
        # Compute maximum allowed acceleration
        a_max = self._compute_max_accel(rho)
        
        return a**2 - a_max**2
        
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of acceleration CBF.
        
        ḣ_accel = 2a_max·d/dt(a_max) - 2a·j
        where j is jerk vector, accounting for dynamics
        """
        # Calculate distance and its rate of change
        dx = human_state['x'] - self.state.x
        dy = human_state['y'] - self.state.y
        rho = np.sqrt(dx**2 + dy**2)
        
        dvx = human_state.get('vx', 0) - self.state.vx
        dvy = human_state.get('vy', 0) - self.state.vy
        rho_dot = (dx * dvx + dy * dvy) / rho
        
        # Current acceleration and jerk
        ax = self.state.get_motor_current(0) * self.state.torque_constant / self.state.mass
        ay = self.state.get_motor_current(1) * self.state.torque_constant / self.state.mass
        a = np.sqrt(ax**2 + ay**2)
        
        # Compute jerk including electrical dynamics
        jx = (self.input_.get_motor_voltage(0) - self.state.get_motor_current(0) * self.state.resistance) \
             / (self.state.inductance * self.state.mass)
        jy = (self.input_.get_motor_voltage(1) - self.state.get_motor_current(1) * self.state.resistance) \
             / (self.state.inductance * self.state.mass)
        
        # Rate of change of maximum acceleration
        a_max = self._compute_max_accel(rho)
        v = np.sqrt(self.state.vx**2 + self.state.vy**2)
        v_dot = (self.state.vx * ax + self.state.vy * ay) / v if v > 0 else 0
        
        # Account for all dynamic effects in a_max derivative
        a_max_dot = (self.a_slope * rho_dot  # Distance effect
                    - a_max * self.velocity_factor * v_dot  # Velocity effect
                    - a_max * self.slip_threshold * self.state.omega_dot)  # Slip effect
        
        return 2 * a_max * a_max_dot - 2 * (ax * jx + ay * jy)
