"""MPC-specific visualization helpers."""

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from matplotlib.patches import Circle, Polygon
from typing import List, Dict, Tuple, Any

class MPCVisualizer:
    """Visualization methods for MPC controller"""

    @staticmethod
    def visualize_mpc_prediction(ax, controller, current_state, predicted_states=None):
        """Visualize MPC predictions.
        
        Args:
            ax: Matplotlib axis to plot on
            controller: MPC controller instance
            current_state: Current robot state
            predicted_states: Precalculated predicted states (optional)
        """
        # If we need to calculate predicted states (this is expensive computation)
        if predicted_states is None:
            predicted_states = MPCVisualizer.get_predicted_trajectory(controller, current_state)
        
        # Create predicted path visualization (if predictions exist)
        if predicted_states is not None and len(predicted_states) > 0:
            # Extract x,y positions
            pred_x = [state[0] for state in predicted_states]
            pred_y = [state[1] for state in predicted_states]
            
            # Draw prediction as a line with markers
            ax.plot(pred_x, pred_y, 'mo-', linewidth=1.5, markersize=4, 
                   label='MPC Prediction', alpha=0.7)
            
            # Draw velocity vectors at prediction points (every other point)
            for i in range(0, len(predicted_states), 2):
                state = predicted_states[i]
                # Scale velocity vector for visualization
                vel_scale = 0.5
                vx, vy = state[3] * vel_scale, state[4] * vel_scale
                
                # Draw a small arrow for velocity direction
                if np.linalg.norm([vx, vy]) > 0.1:  # Only if velocity is significant
                    ax.arrow(state[0], state[1], vx, vy, head_width=0.1, 
                            head_length=0.15, fc='m', ec='m', alpha=0.6)

    @staticmethod
    def visualize_safety_barriers(ax, controller, human_states, config):
        """Visualize safety barriers for MPC.
        
        Args:
            ax: Matplotlib axis to plot on
            controller: MPC controller instance
            human_states: List of human states
            config: Configuration dictionary
        """
        # Visualize CBF safety barriers if available
        if hasattr(controller, 'safety_barriers') and controller.safety_barriers:
            # Get safety parameters from configuration
            rho_0 = config['safety']['rho_0']  # Minimum front distance (m)
            rho_1 = config['safety']['rho_1']  # Minimum side distance (m)
            
            # Draw different barrier regions
            for i, human in enumerate(human_states):
                human_x, human_y = human['x'], human['y']
                
                # Add safety constraint visualization - the barrier boundary 
                # where constraint becomes active
                barrier_circle = Circle((human_x, human_y), rho_0 + 0.1, 
                                      color='orange', alpha=0.2, fill=True,
                                      label='Barrier Boundary' if i==0 else "")
                ax.add_patch(barrier_circle)

    @staticmethod
    def get_predicted_trajectory(controller, current_state, desired_state=None):
        """Get predicted trajectory from MPC controller.
        
        Args:
            controller: MPC controller instance
            current_state: Current robot state
            desired_state: Desired state (optional)
            
        Returns:
            List of predicted states or None if not available
        """
        # Extract MPC prediction if available from controller
        try:
            # First check if controller has directly stored predicted states
            if hasattr(controller, 'predicted_states') and controller.predicted_states is not None:
                return controller.predicted_states
                
            # If not, try to compute them from last_solution
            predicted_states = []
            
            # We need to be careful not to run the expensive optimization again
            # just for visualization. Instead, we extract the last solution.
            if hasattr(controller, 'last_solution') and controller.last_solution is not None:
                # Use controller's prediction horizon parameter
                Hp = controller.params.Hp if hasattr(controller, 'params') else 3
                
                # Initialize with current state
                predicted_states = [current_state]
                
                # Simulate forward dynamics for the prediction horizon using
                # the cached optimal control sequence
                state = current_state.copy()
                
                # Simple kinematics model to predict trajectory
                dt = controller.params.dt if hasattr(controller, 'params') else 0.01
                wheel_radius = controller.config['robot']['wheel_radius']
                wheelbase = controller.config['robot']['wheelbase']
                voltage_factor = controller.config['robot']['motor'].get('voltage_speed_factor', 0.1)
                gear_ratio = controller.config['robot']['motor'].get('gear_ratio', 15.0)
                
                # For each step in control horizon
                for k in range(min(Hp, len(controller.last_solution))):
                    # Extract control action from the cached solution
                    u = controller.last_solution[k]
                    
                    # Control inputs
                    delta_f = u[0]
                    delta_r = u[1]
                    # Average wheel voltage
                    v_wheel = sum(u[2:]) / 4.0  
                    
                    # Calculate vehicle speed
                    v = v_wheel * wheel_radius / (voltage_factor * gear_ratio)
                    v = min(v, 5.0)  # Same limit as in controller
                    
                    # Current state
                    x, y, theta = state[0:3]
                    
                    # Compute omega based on bicycle model
                    omega = v * (np.tan(delta_f) - np.tan(delta_r)) / wheelbase
                    omega = np.clip(omega, -2.0, 2.0)  # Same limit as in controller
                    
                    # Calculate velocity in global frame
                    vx = v * np.cos(theta)
                    vy = v * np.sin(theta)
                    
                    # Updated state with Euler integration
                    x_next = x + vx * dt
                    y_next = y + vy * dt
                    theta_next = theta + omega * dt
                    
                    # Create the next state vector
                    state = np.array([x_next, y_next, theta_next, vx, vy, omega])
                    predicted_states.append(state)
                
                return predicted_states
            return None
        except Exception as e:
            print(f"Warning: Could not extract MPC prediction: {e}")
            return None
