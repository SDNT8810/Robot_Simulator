"""Basic simulation environment."""

import numpy as np
from pathlib import Path
from src.simulation.scenarios import BaseScenario
from src.controllers.mpc import BiLevelMPC
from src.controllers.pid import PID
from src.models.robot import Robot4WSD
from src.safety.barrier import ControlBarrierFunction
from src.safety.barrier import DistanceBarrier, YieldingBarrier, SpeedBarrier, AccelBarrier

class Simulation:
    """Basic simulation class."""
    
    def __init__(self, scenario: BaseScenario):
        """Initialize simulation."""
        config = scenario.config
        self.time = 0.0
        self.log_dt = config['timing']['log_frequency']
        self.scenario = scenario
        self.dt = config['timing']['time_step']
        self.state = scenario.get_initial_state()
        self.desired_state = scenario.get_desired_state(self.time)
        self.duration = scenario.duration
        self.robot = scenario.robot
        self.config = config
        self.action = np.concatenate([[0, 0, 0, 0, 0, 0]])

        
        # Initialize controller based on config
        if config['controller']['type'] == 'PID':
            self.controller = PID(config)
        elif config['controller']['type'] == 'MPC':
            self.controller = BiLevelMPC(config)
        else:
            raise ValueError(f"Unsupported controller type: {config['controller']['type']}")
        
        # Store history for visualization
        self.state_history = {0.0: self.state.copy()}
        self.input_history = {}  # Store control inputs
        
    def step(self) -> bool:
        """Perform one simulation step."""
        if self.scenario is None:
            return False
            
        self.time += self.dt
        self.desired_state = self.scenario.get_desired_state(self.time)
        action = self.controller.action(self.state, self.desired_state)
        
        # Store control inputs for visualization
        self.input_history[self.time] = action.copy()

        # Check if simulation is still running
        if self.scenario.is_running(state=self.state):
            self.state = self.update_state(self.state, action)
            self.state_history[self.time] = self.state.copy()
            return True
        else:
            return False

    def update_state(self, state, action):
        """Update simulation state using full dynamics."""
        new_state = state.copy()
        
        # Extract control inputs
        delta_front, delta_rear = action[0:2]
        wheel_velocities = action[2:6]
        
        # Average wheel velocities for simplified model
        v_wheels = np.mean(wheel_velocities)
        
        # Calculate vehicle velocities based on steering geometry
        v = v_wheels  # Simplified longitudinal velocity
        omega = v * (np.tan(delta_front) - np.tan(delta_rear)) / self.robot.wheelbase
        
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
        alpha = 0.3  # Smoothing factor
        new_state[3] = (1 - alpha) * state[3] + alpha * vx  # vx
        new_state[4] = (1 - alpha) * state[4] + alpha * vy  # vy
        new_state[5] = (1 - alpha) * state[5] + alpha * omega  # omega
        
        return new_state

    def get_state_at(self, time: float) -> np.ndarray:
        """Get interpolated state at specified time."""
        times = np.array(list(self.state_history.keys()))
        idx = np.searchsorted(times, time)
        if idx == 0:
            return self.state_history[times[0]]
        if idx == len(times):
            return self.state_history[times[-1]]
            
        # Interpolate between nearest times
        t0, t1 = times[idx-1], times[idx]
        s0, s1 = self.state_history[t0], self.state_history[t1]
        alpha = (time - t0) / (t1 - t0)
        return s0 * (1 - alpha) + s1 * alpha

    def get_control_inputs(self) -> dict:
        """Get history of control inputs."""
        return self.input_history

    def get_angular_velocities(self) -> np.ndarray:
        """Get array of angular velocities over time.
        
        Returns:
            Array of angular velocities
        """
        times = sorted(self.angular_velocities.keys())
        return np.array([self.angular_velocities[t] for t in times])
