"""Basic robot model."""

import numpy as np
from typing import Dict, Any

class Robot4WSD:
    """Simple robot model class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize robot model."""
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
        self.delta_front = 0.0
        self.delta_rear = 0.0
        
        # Wheel velocities [FL, FR, RL, RR]
        self.wheel_velocities = np.zeros(4)
        
        # Control parameters
        self.dt = config['timing']['time_step']
        self.max_velocity = config['safety']['limits']['velocity']['max']
        self.max_angular_velocity = config['safety']['limits']['velocity']['omega_max']
        self.max_steering_angle = config['controller']['mpc']['constraints']['steering']['max']
        self.config = config
