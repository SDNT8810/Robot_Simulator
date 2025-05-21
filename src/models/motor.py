"""Electric motor model with tire dynamics based on ru-racer formulations."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class MotorState:
    """State of an electric motor."""
    current: float = 0.0      # Motor current [A]
    speed: float = 0.0        # Motor angular speed [rad/s]
    temperature: float = 20.0  # Motor temperature [°C]

class ElectricMotor:
    """Electric motor model with tire dynamics based on ru-racer formulations."""
    
    def __init__(self, config):
        """Initialize motor parameters from config."""
        motor_config = config['robot']['motor']
        
        # Electrical parameters
        self.R = motor_config['resistance']         # Terminal resistance [Ω]
        self.L = motor_config['inductance']         # Terminal inductance [H]
        self.Kt = motor_config['torque_constant']   # Torque constant [Nm/A]
        self.Ke = motor_config['back_emf_constant'] # Back-EMF constant [V/rad/s]
        self.i_max = motor_config['max_current']    # Maximum current [A]
        self.V_nom = motor_config['nominal_voltage'] # Nominal voltage [V]
        
        # Mechanical parameters
        self.J = motor_config['inertia']           # Rotor inertia [kg⋅m²]
        self.b = motor_config['damping']           # Viscous damping [Nm⋅s/rad]
        self.N = motor_config['gear_ratio']        # Gear ratio
        self.eta = motor_config['efficiency']      # Motor efficiency
        
        # Thermal parameters (optional)
        self.T_max = motor_config.get('max_temp', 120.0)     # Maximum temperature [°C]
        self.R_th = motor_config.get('thermal_resistance', 1.0) # Thermal resistance [°C/W]
        self.C_th = motor_config.get('thermal_capacity', 10.0)  # Thermal capacity [J/°C]
        
        # Tire model parameters (from ru-racer Bike.m)
        tire_config = config['robot']['tire']
        self.tire_B = tire_config.get('friction', {}).get('B', 10.275)  # Stiffness factor
        self.tire_C = tire_config.get('friction', {}).get('C', 1.9)     # Shape factor
        self.tire_D = tire_config.get('friction', {}).get('D', 1.0)     # Peak factor
        self.tire_E = tire_config.get('friction', {}).get('E', 0.97)    # Curvature factor
        
        # Wheel parameters
        self.wheel_radius = config['robot']['wheel_radius']
        
        # Initialize state
        self.state = MotorState()
    
    def update(self, voltage: float, wheel_velocity: float, slip_angle: float, normal_force: float, dt: float) -> Tuple[float, float]:
        """
        Update motor state and calculate tire forces based on ru-racer wheel_model.m.
        
        Args:
            voltage: Applied voltage [V]
            wheel_velocity: Linear velocity at tire contact [m/s]
            slip_angle: Slip angle [rad] (lateral)
            normal_force: Normal force on the tire [N]
            dt: Time step [s]
            
        Returns:
            tuple: (longitudinal_force, lateral_force)
        """
        # Limit input voltage to realistic values
        voltage = np.clip(voltage, -self.V_nom, self.V_nom)
        
        # Calculate back-EMF
        v_emf = self.Ke * self.state.speed
        
        # Electrical dynamics (di/dt) - from ru-racer similar to RL circuit equation
        di_dt = (voltage - v_emf - self.R * self.state.current) / self.L
        
        # Update current with Euler integration
        current_next = self.state.current + di_dt * dt
        self.state.current = np.clip(current_next, -self.i_max, self.i_max)
        
        # Calculate motor torque including efficiency
        motor_torque = self.Kt * self.state.current * self.eta
        
        # Calculate wheel angular velocity
        wheel_angular_speed = self.state.speed / self.N
        expected_velocity = wheel_angular_speed * self.wheel_radius
        
        # Calculate longitudinal slip ratio (from ru-racer wheel_model.m)
        if abs(wheel_velocity) < 0.001 and abs(expected_velocity) < 0.001:
            slip_ratio = 0.0
        else:
            slip_ratio = (expected_velocity - wheel_velocity) / max(abs(expected_velocity), abs(wheel_velocity), 0.001)
        
        # FIX: Ensure voltage controls direction correctly by adjusting sign of slip
        # If voltage is positive, we want to go forward (positive slip = wheel turning faster than ground)
        # If voltage is negative, we want to go backward (negative slip = wheel turning slower than ground)
        if voltage > 0 and slip_ratio < 0:
            # If voltage is positive but wheel is slipping backward, adjust slip for proper force direction
            slip_ratio = -slip_ratio
        elif voltage < 0 and slip_ratio > 0:
            # If voltage is negative but wheel is slipping forward, adjust slip for proper force direction
            slip_ratio = -slip_ratio
                
        # Calculate tire forces using Pacejka formula from ru-racer
        F_x, F_y = self.calc_tire_forces(slip_ratio, slip_angle, normal_force)
        
        # FIX: Adjust direction based on voltage sign if the speed is very low (overcome static friction)
        if abs(wheel_angular_speed) < 0.01 and abs(voltage) > 0.2:
            # Add static breakaway force in voltage direction, reduced to avoid excessive force
            static_boost = 0.1 * normal_force * np.sign(voltage)
            F_x += static_boost
        
        # Use a consistent force model without voltage-dependent boosting
        # Convert longitudinal force to torque at the wheel
        wheel_torque = F_x * self.wheel_radius
        
        # Net torque on motor accounting for gear ratio
        net_torque = motor_torque - wheel_torque/self.N - self.b * self.state.speed
        
        # Update motor speed with Euler integration
        speed_next = self.state.speed + (net_torque / self.J) * dt
        self.state.speed = speed_next
        
        # Basic thermal model (optional)
        p_loss = self.R * self.state.current**2  # Power loss
        dT_dt = (p_loss * self.R_th - (self.state.temperature - 20.0)) / self.C_th
        self.state.temperature = np.clip(self.state.temperature + dT_dt * dt, 20.0, self.T_max)
        
        return F_x, F_y
        
    def calc_tire_forces(self, slip_ratio: float, slip_angle: float, normal_force: float) -> Tuple[float, float]:
        """
        Calculate tire forces using Pacejka magic formula from ru-racer wheel_model.m.
        
        Args:
            slip_ratio: Longitudinal slip ratio
            slip_angle: Lateral slip angle [rad]
            normal_force: Normal force on the tire [N]
            
        Returns:
            tuple: (longitudinal_force, lateral_force)
        """
        # Longitudinal force calculation (based on ru-racer wheel_model.m)
        lambda_x = -slip_ratio  # Slip direction convention from ru-racer
        alpha_B_x = np.arcsin(np.sin(lambda_x))  # Handle angle wrapping
        
        # Scale tire parameters for better dynamics model performance
        # These values are modified from the ru-racer model to work better with dynamics
        tire_B = self.tire_B 
        tire_C = self.tire_C  
        tire_D = self.tire_D
        tire_E = self.tire_E
        
        # Pacejka magic formula from ru-racer
        fx_coef = tire_D * np.sin(tire_C * np.arctan(tire_B * (1 - tire_E) * alpha_B_x + 
                                                         tire_E * np.arctan(tire_B * alpha_B_x)))
        F_x = fx_coef * normal_force
        
        # Sign correction (from ru-racer wheel_model.m)
        if F_x * lambda_x > 0:
            F_x = -F_x
            
        # Lateral force calculation
        alpha_B_y = np.arcsin(np.sin(slip_angle))
        fy_coef = tire_D * np.sin(tire_C * np.arctan(tire_B * (1 - tire_E) * alpha_B_y + 
                                                         tire_E * np.arctan(tire_B * alpha_B_y)))
        F_y = fy_coef * normal_force
        
        # Sign correction (from ru-racer wheel_model.m)
        if F_y * slip_angle > 0:
            F_y = -F_y
            
        return F_x, F_y
    
