"""Control Barrier Functions for robot safety.

Implementation based on the Bi-Level Performance-Safety Consideration paper.
CBFs are defined according to equations (7), (8), (9), (10) and condition (11).

Implementation Summary:

1. DistanceBarrier - Equation (7): Maintains safe distance with different thresholds
   for front (ρ_0=2.0m) and side (ρ_1=1.0m) regions

2. YieldingBarrier - Equation (8): Enforces yielding behavior when approaching humans
   Returns approach rate (dρ/dt) in yielding zones

3. SpeedBarrier - Equation (9): Limits speed based on distance to humans
   Uses ν_M(ρ_hi) = V_M · tanh(ρ_hi) for speed limits

4. AccelBarrier - Equation (10): Constrains acceleration near humans
   Uses distance-dependent acceleration limits

All CBFs implement the condition: C_ji = ḣ_ji + α·h²_ji ≥ 0 (Equation 11)
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ControlBarrierFunction(ABC):
    """Base class for all Control Barrier Functions."""
    
    def __init__(self, alpha: float = 1.0):
        """Initialize CBF with class-K function parameter."""
        self.alpha = alpha
        self.robot_state: Optional[Dict[str, float]] = None
    
    def set_robot_state(self, state: Dict[str, float]):
        """Set current robot state."""
        self.robot_state = state
    
    def set_robot_input(self, input_dict: Dict[str, float]):
        """Set current robot input (for compatibility)."""
        pass
    
    def get_adaptive_alpha(self, robot_state: Dict[str, float]) -> float:
        """Get adaptive alpha parameter."""
        return self.alpha
    
    @abstractmethod
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate barrier function h(x)."""
        pass
    
    @abstractmethod
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative ḣ(x)."""
        pass
    
    def constraint_condition(self, human_state: Dict[str, float]) -> float:
        """Evaluate CBF condition: C = ḣ + α·h²."""
        h_val = self.h(human_state)
        h_dot_val = self.h_dot(human_state)
        return h_dot_val + self.alpha * (h_val ** 2)


class DistanceBarrier(ControlBarrierFunction):
    """Distance CBF - Equation (7): Maintains safe distance with different thresholds."""
    
    def __init__(self, config=None, alpha: float = 1.0, rho_0: float = 2.0, rho_1: float = 1.0, theta_0: float = np.pi/4):
        """
        Initialize Distance CBF.
        
        Args:
            config: Configuration dictionary (optional)
            alpha: CBF parameter
            rho_0: Safety threshold for front region (m)
            rho_1: Safety threshold for side region (m)
            theta_0: Critical angular range (rad)
        """
        # If config is provided, extract parameters from it
        if config is not None and isinstance(config, dict):
            safety_config = config.get('safety', {})
            alpha = safety_config.get('cbf_dynamics', {}).get('alpha', 1.0)
            rho_0 = safety_config.get('rho_0', 2.0)
            rho_1 = safety_config.get('rho_1', 1.0) 
            theta_0 = safety_config.get('theta_0', np.pi/4)
        
        super().__init__(alpha)
        self.rho_0 = rho_0
        self.rho_1 = rho_1
        self.theta_0 = theta_0
    
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate distance barrier function - Equation (7)."""
        if self.robot_state is None:
            return 0.0
        
        # Calculate distance to human
        dx = human_state['x'] - self.robot_state['x']
        dy = human_state['y'] - self.robot_state['y']
        rho_hi = np.sqrt(dx**2 + dy**2)
        
        # Calculate angular deviation
        robot_heading = self.robot_state['theta']
        angle_to_human = np.arctan2(dy, dx)
        theta_hi = abs(angle_to_human - robot_heading)
        theta_hi = min(theta_hi, 2*np.pi - theta_hi)  # Normalize to [0, π]
        
        # Apply Equation (7)
        if abs(theta_hi) < self.theta_0:  # Front region
            return rho_hi - self.rho_0
        else:  # Side region
            return rho_hi - self.rho_1
    
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of distance barrier."""
        if self.robot_state is None:
            return 0.0
        
        # Calculate relative position and velocity
        dx = human_state['x'] - self.robot_state['x']
        dy = human_state['y'] - self.robot_state['y']
        dvx = human_state.get('vx', 0.0) - self.robot_state.get('vx', 0.0)
        dvy = human_state.get('vy', 0.0) - self.robot_state.get('vy', 0.0)
        
        rho_hi = np.sqrt(dx**2 + dy**2)
        if rho_hi < 1e-6:
            return 0.0
        
        # Rate of change of distance
        return (dx * dvx + dy * dvy) / rho_hi


class YieldingBarrier(ControlBarrierFunction):
    """Yielding CBF - Equation (8): Enforces yielding behavior when approaching humans."""
    
    def __init__(self, config=None, alpha: float = 1.0, rho_0: float = 2.0, rho_1: float = 1.0, theta_0: float = np.pi/4):
        """
        Initialize Yielding CBF.
        
        Args:
            config: Configuration dictionary (optional)
            alpha: CBF parameter
            rho_0: Front region threshold (m)
            rho_1: Side region threshold (m)
            theta_0: Critical angular range (rad)
        """
        # If config is provided, extract parameters from it
        if config is not None and isinstance(config, dict):
            safety_config = config.get('safety', {})
            alpha = safety_config.get('cbf_dynamics', {}).get('alpha', 1.0)
            rho_0 = safety_config.get('rho_0', 2.0)
            rho_1 = safety_config.get('rho_1', 1.0)
            theta_0 = safety_config.get('theta_0', np.pi/4)
        
        super().__init__(alpha)
        self.rho_0 = rho_0
        self.rho_1 = rho_1
        self.theta_0 = theta_0
    
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate yielding barrier function - Equation (8)."""
        if self.robot_state is None:
            return 0.0
        
        # Calculate distance and angular deviation
        dx = human_state['x'] - self.robot_state['x']
        dy = human_state['y'] - self.robot_state['y']
        rho_hi = np.sqrt(dx**2 + dy**2)
        
        robot_heading = self.robot_state['theta']
        angle_to_human = np.arctan2(dy, dx)
        theta_hi = abs(angle_to_human - robot_heading)
        theta_hi = min(theta_hi, 2*np.pi - theta_hi)
        
        # Check if in yielding zone
        in_front_yield_zone = (rho_hi <= self.rho_0 and abs(theta_hi) < self.theta_0)
        in_side_yield_zone = (rho_hi <= self.rho_1 and abs(theta_hi) >= self.theta_0)
        
        if in_front_yield_zone or in_side_yield_zone:
            # Calculate approach rate (dρ/dt)
            dvx = human_state.get('vx', 0.0) - self.robot_state.get('vx', 0.0)
            dvy = human_state.get('vy', 0.0) - self.robot_state.get('vy', 0.0)
            
            if rho_hi < 1e-6:
                return 0.0
            
            drho_dt = (dx * dvx + dy * dvy) / rho_hi
            return drho_dt  # Should be non-negative (not approaching)
        else:
            return 1.0  # Not in yielding zone, always safe
    
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of yielding barrier."""
        # For yielding barrier, h_dot represents acceleration effects
        return 0.0  # Simplified implementation


class SpeedBarrier(ControlBarrierFunction):
    """Speed CBF - Equation (9): Limits speed based on distance to humans."""
    
    def __init__(self, config=None, alpha: float = 1.0, V_M: float = 2.0):
        """
        Initialize Speed CBF.
        
        Args:
            config: Configuration dictionary (optional)
            alpha: CBF parameter
            V_M: Robot's absolute maximum speed (m/s)
        """
        # If config is provided, extract parameters from it
        if config is not None and isinstance(config, dict):
            safety_config = config.get('safety', {})
            alpha = safety_config.get('cbf_dynamics', {}).get('alpha', 1.0)
            V_M = safety_config.get('limits', {}).get('velocity', {}).get('max', 2.0)
        
        super().__init__(alpha)
        self.V_M = V_M
    
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate speed barrier function - Equation (9)."""
        if self.robot_state is None:
            return 0.0
        
        # Calculate distance to human
        dx = human_state['x'] - self.robot_state['x']
        dy = human_state['y'] - self.robot_state['y']
        rho_hi = np.sqrt(dx**2 + dy**2)
        
        # Calculate maximum permissible speed
        nu_M = self.V_M * np.tanh(rho_hi)
        
        # Current robot speed
        vx = self.robot_state.get('vx', 0.0)
        vy = self.robot_state.get('vy', 0.0)
        v_magnitude = np.sqrt(vx**2 + vy**2)
        
        # h_s,i = ν_M(ρ_hi) - |v|
        return nu_M - v_magnitude
    
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of speed barrier."""
        if self.robot_state is None:
            return 0.0
        
        # Calculate distance and its derivative
        dx = human_state['x'] - self.robot_state['x']
        dy = human_state['y'] - self.robot_state['y']
        rho_hi = np.sqrt(dx**2 + dy**2)
        
        if rho_hi < 1e-6:
            return 0.0
        
        # Rate of change of distance
        dvx = human_state.get('vx', 0.0) - self.robot_state.get('vx', 0.0)
        dvy = human_state.get('vy', 0.0) - self.robot_state.get('vy', 0.0)
        drho_dt = (dx * dvx + dy * dvy) / rho_hi
        
        # Derivative of maximum speed limit
        dnu_M_dt = self.V_M * (1 - np.tanh(rho_hi)**2) * drho_dt
        
        # Derivative of current speed magnitude (simplified)
        return dnu_M_dt  # Neglecting robot acceleration for simplicity


class AccelBarrier(ControlBarrierFunction):
    """Acceleration CBF - Equation (10): Constrains acceleration near humans."""
    
    def __init__(self, config=None, alpha: float = 1.0, a_max_base: float = 1.0):
        """
        Initialize Acceleration CBF.
        
        Args:
            config: Configuration dictionary (optional)
            alpha: CBF parameter
            a_max_base: Base maximum acceleration (m/s²)
        """
        # If config is provided, extract parameters from it
        if config is not None and isinstance(config, dict):
            safety_config = config.get('safety', {})
            alpha = safety_config.get('cbf_dynamics', {}).get('alpha', 1.0)
            a_max_base = safety_config.get('limits', {}).get('acceleration', {}).get('base', 1.0)
        
        super().__init__(alpha)
        self.a_max_base = a_max_base
    
    def h(self, human_state: Dict[str, float]) -> float:
        """Evaluate acceleration barrier function - Equation (10)."""
        if self.robot_state is None:
            return 0.0
        
        # Calculate distance to human
        dx = human_state['x'] - self.robot_state['x']
        dy = human_state['y'] - self.robot_state['y']
        rho_hi = np.sqrt(dx**2 + dy**2)
        
        # Distance-dependent maximum acceleration
        a_max_rho = self.a_max_base * np.tanh(rho_hi)
        
        # Current acceleration magnitude (simplified estimation)
        ax = self.robot_state.get('ax', 0.0)
        ay = self.robot_state.get('ay', 0.0)
        a_magnitude = np.sqrt(ax**2 + ay**2)
        
        # h_a,i = a_max(ρ_hi) - d|v|/dt
        return a_max_rho - a_magnitude
    
    def h_dot(self, human_state: Dict[str, float]) -> float:
        """Evaluate time derivative of acceleration barrier."""
        # Simplified implementation
        return 0.0


def extract_safety_data(sim) -> Dict[str, Any]:
    """Extract safety constraint data from simulation."""
    safety_data: Dict[str, Any] = {
        'num_humans': 0
    }
    
    try:
        controller = sim.controller
        state = sim.robot.state
        
        if not hasattr(controller, 'safety_barriers'):
            return safety_data
        
        robot_state = {
            'x': float(state[0]), 'y': float(state[1]), 'theta': float(state[2]),
            'vx': float(state[3]), 'vy': float(state[4]), 'omega': float(state[5])
        }
        
        # Extract human states
        human_states = []
        if hasattr(sim.scenario, 'config') and sim.scenario.config:
            scenario_config = sim.scenario.config.get('scenario', {}).get(sim.scenario.scenario_name, {})
            if 'humans' in scenario_config:
                positions = scenario_config['humans'].get('positions', [])
                velocities = scenario_config['humans'].get('velocities', [])
                for pos, vel in zip(positions, velocities):
                    human_states.append({
                        'x': float(pos[0]), 'y': float(pos[1]),
                        'vx': float(vel[0]), 'vy': float(vel[1])
                    })
        
        safety_data['num_humans'] = len(human_states)
        
        # Get all 4 barriers
        barrier_names = ['distance', 'yielding', 'speed', 'accel']
        barriers = {}
        for i, barrier in enumerate(controller.safety_barriers):
            if i < len(barrier_names):
                barriers[barrier_names[i]] = barrier
                barrier.set_robot_state(robot_state)
        
        # Initialize aggregated safety fields for backward compatibility
        for barrier_name in barrier_names:
            safety_data[f'h_{barrier_name}'] = 0.0
            safety_data[f'h_dot_{barrier_name}'] = 0.0
            safety_data[f'cbf_{barrier_name}'] = 0.0
            safety_data[f'violation_{barrier_name}'] = False
            safety_data[f'alpha_{barrier_name}'] = 1.0
        
        # Calculate all 4 barriers for each human individually
        min_values = {name: float('inf') for name in barrier_names}
        max_violations = {name: False for name in barrier_names}
        
        for human_idx, human_state in enumerate(human_states):
            for barrier_name in barrier_names:
                if barrier_name in barriers:
                    barrier = barriers[barrier_name]
                    
                    h_val = barrier.h(human_state)
                    h_dot_val = barrier.h_dot(human_state)
                    cbf_val = barrier.constraint_condition(human_state)
                    alpha_val = barrier.get_adaptive_alpha(robot_state)
                    
                    # Create flat field names for CSV: h_distance_human0, h_yielding_human1, etc.
                    safety_data[f'h_{barrier_name}_human{human_idx}'] = h_val
                    safety_data[f'h_dot_{barrier_name}_human{human_idx}'] = h_dot_val
                    safety_data[f'cbf_{barrier_name}_human{human_idx}'] = cbf_val
                    safety_data[f'violation_{barrier_name}_human{human_idx}'] = cbf_val < 0
                    safety_data[f'alpha_{barrier_name}_human{human_idx}'] = alpha_val
                    
                    # Track minimum values for aggregated fields (most critical)
                    if h_val < min_values[barrier_name]:
                        min_values[barrier_name] = h_val
                        safety_data[f'h_{barrier_name}'] = h_val
                        safety_data[f'h_dot_{barrier_name}'] = h_dot_val
                        safety_data[f'cbf_{barrier_name}'] = cbf_val
                        safety_data[f'alpha_{barrier_name}'] = alpha_val
                    
                    # Track violations for aggregated fields
                    if cbf_val < 0:
                        max_violations[barrier_name] = True
        
        # Set aggregated violation flags
        for barrier_name in barrier_names:
            safety_data[f'violation_{barrier_name}'] = max_violations[barrier_name]

    except Exception as e:
        logger.info(f"Error extracting safety data: {e}")
        
    return safety_data