"""Real-time visualization for the robot simulation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Polygon, Circle
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict
from src.simulation.simulator import Simulation
from src.models.robot import Robot4WSD
from src.utils.mpc_visualization import MPCVisualizer
import math

class RobotVisualizer:
    """Real-time visualization of the robot state."""
    
    @classmethod
    def plot_results(cls, simulation: Simulation):
        """Plot simulation results."""
        # Get current state and inputs
        state = simulation.robot.state
        x, y, theta = state[0:3]
        vx, vy, omega = state[3:6]

        # Get current control inputs
        current_input = simulation.input_history.get(simulation.time, np.zeros(6))
        delta_front, delta_rear = current_input[0:2]
        wheel_velocities = current_input[2:6]
        
        # Get current desired state for visualization
        desired_state = simulation.desired_state
        x_d, y_d = desired_state[0:2]
        vx_d, vy_d = desired_state[3:5]
        
        # Create figure if it doesn't exist
        if not hasattr(cls, 'fig'):
            cls.fig = plt.figure(figsize=(16, 12))  # Increased height for flexible layout
            mng = plt.get_current_fig_manager()
            
            # Try to make window full screen in a cross-platform way
            try:
                manager = plt.get_current_fig_manager()
                if hasattr(manager, 'window'):
                    manager.window.showMaximized()  # Qt backend
                elif hasattr(manager, 'frame'):
                    manager.frame.Maximize(True)    # WX backend
                elif hasattr(manager, 'full_screen_toggle'):
                    manager.full_screen_toggle()    # GTK backend
                elif hasattr(manager, 'resize'):
                    # Get screen dimensions using tkinter as fallback
                    import tkinter
                    root = tkinter.Tk()
                    manager.resize(root.winfo_screenwidth(), root.winfo_screenheight())
                    root.destroy()
            except Exception:
                # If all fails, just use the default size
                pass
            
            # Read plot configuration from config
            cls.enabled_plots = cls._get_enabled_plots(simulation.config)
            
            # Create dynamic layout based on enabled plots
            cls._create_layout(simulation.config)
            
            # For compatibility with existing code
            # Map is now always included in enabled_plots, so this should always work
            cls.ax_robot = cls.ax_plots['Map']
            
            # For other plots, set if available
            cls.ax_states = cls.ax_plots.get('State', None)
            cls.ax_v_omega = cls.ax_plots.get('Velocity', None)
            
            cls.setup_plots()
            # Initialize history with additional fields for Error and Control plots
            cls.history = {
                't': [], 'x': [], 'y': [], 'v': [], 'omega': [],
                'theta': [], 'vx': [], 'vy': [], 'x_d': [], 'y_d': [], 
                'vx_d': [], 'vy_d': [], 'omega_d': [], 'theta_d': [],
                'delta_front': [], 'delta_rear': [], 'V_FL': [], 'V_FR': [], 
                'V_RL': [], 'V_RR': []
            }
            cls.robot_patches = []
            cls.path_line = None
            cls.state_lines = {}
            cls.v_omega_lines = {}
            cls.error_lines = {}
            cls.control_lines = {}
            
            # Plot start and goal positions
            start_pos = simulation.scenario.get_initial_state()[:2]
            if simulation.scenario.scenario_name == 'to_goal':
                goal_pos = simulation.scenario.goal[:2]
            else:  # circle scenario
                goal_pos = simulation.scenario.center
                
            cls.ax_robot.plot(start_pos[0], start_pos[1], 'g*', markersize=15, label='Start', zorder=5)
            cls.ax_robot.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Target', zorder=5)
            
            # Add humans to the plot
            if simulation.scenario.scenario_name == 'to_goal' and 'humans' in simulation.scenario.config['scenario']['to_goal']:
                cls._update_human_visualization(simulation)
                    
            # Create desired circle path if applicable
            if simulation.scenario.scenario_name == 'circle':
                radius = simulation.scenario.radius
                center = simulation.scenario.center
                t = np.linspace(0, 2*np.pi, 100)
                x_circle = center[0] + radius * np.cos(t)
                y_circle = center[1] + radius * np.sin(t)
                cls.ax_robot.plot(x_circle, y_circle, 'g-.', label='Desired', linewidth=2)
            
            # Initialize desired position marker
            cls.desired_pos_marker = None
            # Initialize velocity vector markers
            cls.actual_vel_arrow = None
            cls.desired_vel_arrow = None
                
        # Calculate theta_d (desired orientation) and omega_d
        if simulation.scenario.scenario_name == 'circle':
            # For circular path, point tangent to the circle
            center = np.array(simulation.scenario.center)
            pos = np.array([x, y])
            direction = pos - center
            # Perpendicular direction (tangent to circle, counterclockwise)
            theta_d = np.arctan2(-direction[0], direction[1])
        else:
            # For point-to-point, point toward the goal
            dx, dy = x_d - x, y_d - y
            theta_d = np.arctan2(dy, dx)
        
        # Get desired omega from desired state
        omega_d = desired_state[5] if len(desired_state) > 5 else 0.0
        
        # Store history including control inputs and desired states
        cls.history['t'].append(simulation.time)
        cls.history['x'].append(x)
        cls.history['y'].append(y)
        cls.history['theta'].append(theta)
        cls.history['vx'].append(vx)
        cls.history['vy'].append(vy)
        cls.history['omega'].append(omega)
        cls.history['v'].append(np.sqrt(vx**2 + vy**2))
        cls.history['x_d'].append(x_d)
        cls.history['y_d'].append(y_d)
        cls.history['vx_d'].append(vx_d)
        cls.history['vy_d'].append(vy_d)
        cls.history['omega_d'].append(omega_d)
        cls.history['theta_d'].append(theta_d)
        
        # Store control inputs
        cls.history['delta_front'].append(delta_front)
        cls.history['delta_rear'].append(delta_rear)
        cls.history['V_FL'].append(wheel_velocities[0])
        cls.history['V_FR'].append(wheel_velocities[1])
        cls.history['V_RL'].append(wheel_velocities[2])
        cls.history['V_RR'].append(wheel_velocities[3])

        # Clear previous robot visualization
        for patch in cls.robot_patches:
            patch.remove()
        cls.robot_patches.clear()
        
        # Clear previous human visualization elements
        if hasattr(cls, 'human_elements'):
            for element in cls.human_elements:
                element.remove()
            cls.human_elements.clear()
        
        # Update Map visualization if enabled
        if 'Map' in cls.enabled_plots:
            # Plot start and goal positions if they haven't been plotted yet
            if not hasattr(cls, 'start_marker'):
                start_pos = simulation.scenario.get_initial_state()[:2]
                if simulation.scenario.scenario_name == 'to_goal':
                    goal_pos = simulation.scenario.goal[:2]
                else:  # circle scenario
                    goal_pos = simulation.scenario.center
                    
                cls.start_marker = cls.ax_plots['Map'].plot(start_pos[0], start_pos[1], 'g*', markersize=15, label='Start', zorder=5)[0]
                cls.goal_marker = cls.ax_plots['Map'].plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Target', zorder=5)[0]
                
                # Create desired circle path if applicable
                if simulation.scenario.scenario_name == 'circle':
                    radius = simulation.scenario.radius
                    center = simulation.scenario.center
                    t = np.linspace(0, 2*np.pi, 100)
                    x_circle = center[0] + radius * np.cos(t)
                    y_circle = center[1] + radius * np.sin(t)
                    cls.circle_path = cls.ax_plots['Map'].plot(x_circle, y_circle, 'g-.', label='Desired', linewidth=2)[0]
            
            # Update robot visualization with steering angles
            cls.update_robot(x, y, theta, simulation.robot, delta_front, delta_rear, wheel_velocities)
            
            # Update human visualization if applicable
            if simulation.scenario.scenario_name == 'to_goal' and 'humans' in simulation.scenario.config['scenario']['to_goal']:
                cls._update_human_visualization(simulation)
            
            # Update path line
            if cls.path_line is None:
                cls.path_line, = cls.ax_plots['Map'].plot(cls.history['x'], cls.history['y'], 'b-', alpha=0.5, label='Actual')
            else:
                cls.path_line.set_data(cls.history['x'], cls.history['y'])
            
            # Clear and redraw the desired position marker (red point)
            if hasattr(cls, 'desired_pos_marker') and cls.desired_pos_marker is not None:
                cls.desired_pos_marker.remove()
            cls.desired_pos_marker = cls.ax_plots['Map'].plot(x_d, y_d, 'ro', markersize=12, label='Desired Pos')[0]
            
            # Draw velocity vectors for better visualization (scale them for visibility)
            vel_scale = 2.0  # Scale factor for visibility
            
            # Clear previous velocity vectors
            if hasattr(cls, 'actual_vel_arrow') and cls.actual_vel_arrow is not None:
                cls.actual_vel_arrow.remove()
            if hasattr(cls, 'desired_vel_arrow') and cls.desired_vel_arrow is not None:
                cls.desired_vel_arrow.remove()
                
            # Draw actual velocity vector (blue)
            cls.actual_vel_arrow = cls.ax_plots['Map'].arrow(
                x, y, vx * vel_scale, vy * vel_scale, 
                head_width=0.2, head_length=0.3, fc='blue', ec='blue', 
                alpha=0.7, label='Actual Vel'
            )
            
            # Draw desired velocity vector (red)
            cls.desired_vel_arrow = cls.ax_plots['Map'].arrow(
                x_d, y_d, vx_d * vel_scale, vy_d * vel_scale, 
                head_width=0.2, head_length=0.3, fc='red', ec='red', 
                alpha=0.7, label='Desired Vel'
            )
        
        # Update State plot if enabled
        if 'State' in cls.enabled_plots:
            # Define state variables to plot
            state_vars = {
                'vx': ['m-', 'Vx [m/s]'],
                'vy': ['c-', 'Vy [m/s]'],
                'omega': ['y-', 'ω [rad/s]']
            }
            
            for var, (style, label) in state_vars.items():
                if var not in cls.state_lines:
                    cls.state_lines[var], = cls.ax_plots['State'].plot(cls.history['t'], 
                                                                cls.history[var], 
                                                                style, 
                                                                label=label)
                else:
                    cls.state_lines[var].set_data(cls.history['t'], cls.history[var])
        
        # Update Velocity plot if enabled
        if 'Velocity' in cls.enabled_plots:
            if 'v' not in cls.v_omega_lines:
                cls.v_omega_lines['v'], = cls.ax_plots['Velocity'].plot(cls.history['t'], 
                                                                    cls.history['v'], 
                                                                    'g-', 
                                                                    label='v [m/s]')
                cls.v_omega_lines['omega'], = cls.ax_plots['Velocity'].plot(cls.history['t'], 
                                                                        cls.history['omega'], 
                                                                        'r-', 
                                                                        label='ω [rad/s]')
            else:
                cls.v_omega_lines['v'].set_data(cls.history['t'], cls.history['v'])
                cls.v_omega_lines['omega'].set_data(cls.history['t'], cls.history['omega'])
        
        # Update Error plot if enabled
        if 'Error' in cls.enabled_plots:
            error_vars = {
                'x': ['r-', 'x error [m]'],
                'y': ['b-', 'y error [m]'],
                'theta': ['g-', 'θ error [rad]'],
                'vx': ['m--', 'vx error [m/s]'],
                'vy': ['c--', 'vy error [m/s]'],
                'omega': ['y--', 'ω error [rad/s]']
            }
            
            # Calculate errors
            x_error = cls.history['x'][-1] - cls.history['x_d'][-1]
            y_error = cls.history['y'][-1] - cls.history['y_d'][-1]
            theta_error = cls.history['theta'][-1] - cls.history['theta_d'][-1]
            vx_error = cls.history['vx'][-1] - cls.history['vx_d'][-1] 
            vy_error = cls.history['vy'][-1] - cls.history['vy_d'][-1]
            omega_error = cls.history['omega'][-1] - cls.history['omega_d'][-1]
            
            # Store errors
            if 'x_error' not in cls.history:
                cls.history['x_error'] = []
                cls.history['y_error'] = []
                cls.history['theta_error'] = []
                cls.history['vx_error'] = []
                cls.history['vy_error'] = []
                cls.history['omega_error'] = []
            
            cls.history['x_error'].append(x_error)
            cls.history['y_error'].append(y_error)
            cls.history['theta_error'].append(theta_error)
            cls.history['vx_error'].append(vx_error)
            cls.history['vy_error'].append(vy_error)
            cls.history['omega_error'].append(omega_error)
            
            # Plot errors
            for var, (style, label) in error_vars.items():
                var_name = f"{var}_error"
                if var_name not in cls.error_lines:
                    cls.error_lines[var_name], = cls.ax_plots['Error'].plot(cls.history['t'], 
                                                                        cls.history[var_name], 
                                                                        style, 
                                                                        label=label)
                else:
                    cls.error_lines[var_name].set_data(cls.history['t'], cls.history[var_name])
        
        # Update ControlInput plot if enabled
        if 'ControlInput' in cls.enabled_plots:
            control_vars = {
                # Steering angles on left y-axis
                'steering': [
                    ('delta_front', 'b-', 'δ front [rad]'),
                    ('delta_rear', 'g-', 'δ rear [rad]')
                ],
                # Voltages on right y-axis
                'voltage': [
                    ('V_FL', 'r--', 'V FL [V]'),
                    ('V_FR', 'm--', 'V FR [V]'),
                    ('V_RL', 'c--', 'V RL [V]'),
                    ('V_RR', 'y--', 'V RR [V]')
                ]
            }
            
            # Plot steering angles on the left y-axis
            for var, style, label in control_vars['steering']:
                if var not in cls.control_lines:
                    cls.control_lines[var], = cls.ax_plots['ControlInput'].plot(
                        cls.history['t'], cls.history[var], style, label=label)
                else:
                    cls.control_lines[var].set_data(cls.history['t'], cls.history[var])
            
            # Create right axis for voltages if it doesn't exist
            if not hasattr(cls, 'ax_voltage'):
                cls.ax_voltage = cls.ax_plots['ControlInput'].twinx()
                cls.ax_voltage.set_ylabel('Voltage [V]', color='r')
                cls.ax_voltage.tick_params(axis='y', labelcolor='r')
            
            # Plot voltages on the right y-axis
            for var, style, label in control_vars['voltage']:
                if var not in cls.control_lines:
                    cls.control_lines[var], = cls.ax_voltage.plot(
                        cls.history['t'], cls.history[var], style, label=label)
                else:
                    cls.control_lines[var].set_data(cls.history['t'], cls.history[var])
        
        # Update SafetyViolation plot if enabled
        if 'SafetyViolation' in cls.enabled_plots:
            # Implementation for safety violation plot would go here
            pass
        
        # Update limits for all plots
        window = simulation.scenario.config['visualization']['view_time_window']
        t_min = max(0, simulation.time - window)
        t_max = simulation.time + 0.1
        
        # Set time limits for time-series plots
        for plot_name, ax in cls.ax_plots.items():
            if plot_name != 'Map':  # Map is not a time-series plot
                ax.set_xlim(t_min, t_max)
        
        # Set voltage axis limits if it exists
        if hasattr(cls, 'ax_voltage'):
            cls.ax_voltage.set_xlim(t_min, t_max)
        
        # Update the Map plot limits
        if 'Map' in cls.enabled_plots:
            margin = simulation.scenario.config['visualization']['margins']['margin']
            if simulation.scenario.config['visualization']['margins']['mode'] == 'Auto_Center':
                cls.ax_plots['Map'].set_xlim(x - margin, x + margin)
                cls.ax_plots['Map'].set_ylim(y - margin, y + margin)
            else:
                x_center, y_center = simulation.config['scenario']['circle']['center']
                cls.ax_plots['Map'].set_xlim(x_center - margin, x_center + margin)
                cls.ax_plots['Map'].set_ylim(y_center - margin, y_center + margin)
        
        # Auto-scale other plots
        if 'State' in cls.enabled_plots:
            state_vars = ['vx', 'vy', 'omega']
            cls._auto_scale_plot('State', simulation, cls.state_lines, state_vars)
        
        if 'Velocity' in cls.enabled_plots:
            velocity_vars = ['v', 'omega']
            cls._auto_scale_plot('Velocity', simulation, cls.v_omega_lines, velocity_vars)
        
        if 'Error' in cls.enabled_plots:
            error_vars = ['x_error', 'y_error', 'theta_error', 'vx_error', 'vy_error', 'omega_error']
            cls._auto_scale_plot('Error', simulation, cls.error_lines, error_vars)
        
        if 'ControlInput' in cls.enabled_plots:
            # Auto-scale steering angles (left y-axis)
            steering_vals = (cls.history['delta_front'][-50:] if cls.history['delta_front'] else []) + \
                         (cls.history['delta_rear'][-50:] if cls.history['delta_rear'] else [])
            if steering_vals:
                plot_margin = simulation.scenario.config['visualization']['plot_margin']
                y_min = min(steering_vals) - 0.1
                y_max = max(steering_vals) + 0.1
                cls.ax_plots['ControlInput'].set_ylim(y_min, y_max)
            
            # Auto-scale voltages (right y-axis)
            if hasattr(cls, 'ax_voltage'):
                voltage_vals = []
                for var in ['V_FL', 'V_FR', 'V_RL', 'V_RR']:
                    if cls.history[var]:
                        voltage_vals.extend(cls.history[var][-50:])
                
                if voltage_vals:
                    plot_margin = 1.0  # Voltage margin
                    y_min = min(voltage_vals) - plot_margin
                    y_max = max(voltage_vals) + plot_margin
                    cls.ax_voltage.set_ylim(y_min, y_max)
        
        # Update legends for all plots
        for plot_name, ax in cls.ax_plots.items():
            if plot_name == 'Map':
                # Map plot has many elements, create a comprehensive legend
                cls._update_map_legend(simulation)
            elif plot_name == 'ControlInput':
                # Special handling for dual y-axis plot
                lines = []
                labels = []
                
                # Get lines and labels from left y-axis
                for l in cls.ax_plots[plot_name].get_lines():
                    lines.append(l)
                    labels.append(l.get_label())
                
                # Get lines and labels from right y-axis (voltages)
                if hasattr(cls, 'ax_voltage'):
                    for l in cls.ax_voltage.get_lines():
                        lines.append(l)
                        labels.append(l.get_label())
                
                # Create a single legend for both axes
                if lines and not hasattr(cls, f'{plot_name}_legend'):
                    setattr(cls, f'{plot_name}_legend', ax.legend(lines, labels, loc='upper right'))
            else:
                # Standard legend for other plots
                ax.legend(loc='upper right')
        
        # Add MPC prediction visualization if controller is MPC
        if 'Map' in cls.enabled_plots and hasattr(simulation, 'controller') and (simulation.config['controller']['mode'] == 'MPC'):
            cls._update_mpc_visualization(simulation)
        
        cls.fig.canvas.draw()
        cls.fig.canvas.flush_events()
    
    @classmethod
    def _get_enabled_plots(cls, config: Dict) -> List[str]:
        """Get list of enabled plot types from configuration."""
        enabled_plots = []
        
        # Get plot configuration
        plot_config = config.get('visualization', {}).get('plots', [])
        
        # Parse the configuration and get enabled plots
        for plot_item in plot_config:
            for plot_name, enabled in plot_item.items():
                # Always include Map and include other plots if they're enabled
                if plot_name == 'Map' or enabled:
                    enabled_plots.append(plot_name)
        
        # Ensure Map is included if not already
        if 'Map' not in enabled_plots:
            enabled_plots.insert(0, 'Map')  # Add Map as the first element
        
        return enabled_plots
    
    @classmethod
    def _create_layout(cls, config: Dict):
        """Create dynamic plot layout based on enabled plots."""
        # Count enabled plots
        num_plots = len(cls.enabled_plots)
        
        # Initialize axes dictionary
        cls.ax_plots = {}
        
        if num_plots == 0:
            # No plots enabled, create a dummy plot
            cls.ax_plots['dummy'] = cls.fig.add_subplot(1, 1, 1)
            cls.ax_plots['dummy'].set_visible(False)
        elif num_plots == 1:
            # Only one plot, use the full figure
            cls.ax_plots[cls.enabled_plots[0]] = cls.fig.add_subplot(1, 1, 1)
        else:
            # Calculate grid dimensions to ensure columns ≥ rows
            # Start with a square-ish grid
            num_cols = math.ceil(math.sqrt(num_plots))
            num_rows = math.ceil(num_plots / num_cols)
            
            # Ensure cols ≥ rows by swapping if needed
            if num_cols < num_rows:
                num_cols, num_rows = num_rows, num_cols
                
            # Create grid
            gs = cls.fig.add_gridspec(num_rows, num_cols)
            
            # Add plots to grid
            for i, plot_name in enumerate(cls.enabled_plots):
                row = i // num_cols
                col = i % num_cols
                cls.ax_plots[plot_name] = cls.fig.add_subplot(gs[row, col])
    
    @classmethod
    def setup_plots(cls):
        """Initialize plot layout."""
        # Setup each plot based on its type
        for plot_name, ax in cls.ax_plots.items():
            if plot_name == 'Map':
                # Robot visualization plot
                ax.set_aspect('equal')
                ax.grid(True)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_title('Position')
            elif plot_name == 'State':
                # State plot (velocity states only)
                ax.grid(True)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Velocity States')
                ax.set_title('Robot Velocity States')
            elif plot_name == 'Velocity':
                # v/omega plot
                ax.grid(True)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Values')
                ax.set_title('Speed and Angular Velocity')
            elif plot_name == 'Error':
                # Error plot
                ax.grid(True)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Error Values')
                ax.set_title('Tracking Errors')
            elif plot_name == 'ControlInput':
                # Control input plot
                ax.grid(True)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Steering Angle [rad]')
                ax.set_title('Control Inputs')
            elif plot_name == 'SafetyViolation':
                # Safety violation plot
                ax.grid(True)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Violation')
                ax.set_title('Safety Violations')
        
        # Initialize visualization elements
        cls.human_elements = []
        cls.mpc_elements = []
        
        plt.tight_layout()
    
    @classmethod
    def update_robot(cls, x: float, y: float, theta: float, robot: 'Robot4WSD', 
                    delta_front: float, delta_rear: float, wheel_velocities: np.ndarray):
        """Draw robot with steered wheels."""
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Draw body
        length = robot.wheelbase
        width = robot.track_width
        body_points = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2],
            [-length/2, -width/2]
        ])
        points = np.dot(body_points, R.T) + np.array([x, y])
        line, = cls.ax_plots['Map'].plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
        cls.robot_patches.append(line)
        
        # Draw direction arrow
        arrow = Arrow(x, y, length/2*c, length/2*s, width=0.3, color='red')
        cls.ax_plots['Map'].add_patch(arrow)
        cls.robot_patches.append(arrow)

        # Draw wheels with steering
        wheel_width = width * 0.2
        wheel_length = length * 0.15
        
        # Wheel positions relative to center (corrected order)
        wheel_positions = [
            [length/2, width/2],    # Front left
            [length/2, -width/2],   # Front right
            [-length/2, width/2],   # Rear left
            [-length/2, -width/2],  # Rear right
        ]
        
        # Steering angles for each wheel (removed negative signs)
        steering_angles = [
            delta_front,  # Front left
            delta_front,  # Front right
            delta_rear,   # Rear left
            delta_rear    # Rear right
        ]
        
        for (wx, wy), delta, v in zip(wheel_positions, steering_angles, wheel_velocities):
            # Transform wheel center position
            wheel_pos = np.dot([wx, wy], R.T) + [x, y]
            
            # Create rotated wheel shape
            wheel_points = np.array([
                [-wheel_length/2, -wheel_width/2],
                [wheel_length/2, -wheel_width/2],
                [wheel_length/2, wheel_width/2],
                [-wheel_length/2, wheel_width/2]
            ])
            
            # Apply steering angle
            c_wheel, s_wheel = np.cos(theta + delta), np.sin(theta + delta)
            R_wheel = np.array([[c_wheel, -s_wheel], [s_wheel, c_wheel]])
            wheel_points = np.dot(wheel_points, R_wheel.T) + wheel_pos
            
            # Draw wheel
            wheel = Polygon(wheel_points, facecolor='blue' if v > 0 else 'red', 
                          alpha=0.5, edgecolor='black')
            cls.ax_plots['Map'].add_patch(wheel)
            cls.robot_patches.append(wheel)
            
            # Add direction indicator on wheel
            indicator_length = wheel_length * 0.8
            indicator_start = wheel_pos
            indicator_end = wheel_pos + indicator_length * np.array([c_wheel, s_wheel])
            indicator, = cls.ax_plots['Map'].plot([indicator_start[0], indicator_end[0]], 
                                               [indicator_start[1], indicator_end[1]], 
                                               'k-', linewidth=1)
            cls.robot_patches.append(indicator)
    
    @classmethod
    def _update_human_visualization(cls, simulation):
        """Update human visualization elements."""
        human_positions = simulation.scenario.config['scenario']['to_goal']['humans'].get('positions', [])
        
        # Initialize human elements list if not exists
        if not hasattr(cls, 'human_elements'):
            cls.human_elements = []
            
        for i, pos in enumerate(human_positions):
            # Plot human as a circle with cross
            human_x, human_y = pos[0], pos[1]
            human_radius = 0.5  # Human radius in meters
            
            # Add human circle
            human_circle = Circle((human_x, human_y), human_radius, 
                                 color='green', alpha=0.5)
            cls.ax_plots['Map'].add_patch(human_circle)
            cls.human_elements.append(human_circle)
            
            # Add cross inside the circle for better visibility
            line1, = cls.ax_plots['Map'].plot([human_x-human_radius/2, human_x+human_radius/2], 
                                            [human_y, human_y], 'k-', linewidth=2)
            line2, = cls.ax_plots['Map'].plot([human_x, human_x], 
                                            [human_y-human_radius/2, human_y+human_radius/2], 
                                            'k-', linewidth=2)
            cls.human_elements.extend([line1, line2])
            
            # Add safety zones
            rho_0 = simulation.scenario.config['safety']['rho_0']  # Front distance
            rho_1 = simulation.scenario.config['safety']['rho_1']  # Side distance
            
            # Safety circles
            front_circle = Circle((human_x, human_y), rho_0, 
                                 color='red', alpha=0.2, fill=False, linestyle='--')
            side_circle = Circle((human_x, human_y), rho_1, 
                                color='green', alpha=0.2, fill=False, linestyle='--')
            cls.ax_plots['Map'].add_patch(front_circle)
            cls.ax_plots['Map'].add_patch(side_circle)
            cls.human_elements.extend([front_circle, side_circle])
            
            # Human ID text
            text = cls.ax_plots['Map'].text(human_x, human_y+human_radius+0.2, f"Human {i+1}", 
                                          horizontalalignment='center', color='black')
            cls.human_elements.append(text)
    
    @classmethod
    def _update_mpc_visualization(cls, simulation):
        """Update MPC prediction visualization."""
        # Clear previous barrier and prediction visualizations
        if hasattr(cls, 'mpc_elements'):
            for element in cls.mpc_elements:
                if element is not None:
                    try:
                        element.remove()
                    except:
                        pass
            cls.mpc_elements = []
        else:
            cls.mpc_elements = []
        
        # Get human states for barrier visualization
        human_states = []
        if 'scenario' in simulation.config and 'to_goal' in simulation.config['scenario'] and 'humans' in simulation.config['scenario']['to_goal']:
            for i, pos in enumerate(simulation.config['scenario']['to_goal']['humans'].get('positions', [])):
                human_states.append({
                    'x': pos[0], 
                    'y': pos[1],
                    'theta': pos[2] if len(pos) > 2 else 0.0
                })
        
        try:    
            # Visualize MPC prediction
            predicted_states = MPCVisualizer.get_predicted_trajectory(
                simulation.controller, 
                simulation.robot.state,
                simulation.desired_state
            )
            
            if predicted_states is not None and len(predicted_states) > 0:
                # Extract x,y positions
                pred_x = [state[0] for state in predicted_states]
                pred_y = [state[1] for state in predicted_states]
                
                # Draw prediction as a line with markers
                prediction_line, = cls.ax_plots['Map'].plot(pred_x, pred_y, 'mo-', linewidth=1.5, 
                                                     markersize=4, label='MPC Prediction', alpha=0.7)
                cls.mpc_elements.append(prediction_line)
        except Exception as e:
            print(f"Warning: Failed to visualize MPC prediction: {e}")
            
        # Visualize safety barriers if available
        try:
            if human_states:
                for i, human in enumerate(human_states):
                    human_x, human_y = human['x'], human['y']
                    
                    # Add safety constraint visualization
                    rho_0 = simulation.config['safety']['rho_0']
                    barrier_circle = Circle((human_x, human_y), rho_0 + 0.1, 
                                       color='orange', alpha=0.2, fill=True,
                                       label='Barrier Boundary' if i==0 else "")
                    cls.ax_plots['Map'].add_patch(barrier_circle)
                    cls.mpc_elements.append(barrier_circle)
        except Exception as e:
            print(f"Warning: Failed to visualize safety barriers: {e}")
            
    @classmethod
    def _update_map_legend(cls, simulation):
        """Create a comprehensive legend for the map plot."""
        if not hasattr(cls, 'path_line') or cls.path_line is None:
            return
            
        # Collect handles and labels for the legend
        handles = []
        labels = []
        
        # Add path line
        handles.append(cls.path_line)
        labels.append('Actual Path')
        
        # Add desired position marker if available
        if hasattr(cls, 'desired_pos_marker') and cls.desired_pos_marker is not None:
            handles.append(cls.desired_pos_marker)
            labels.append('Desired Pos')
        
        # Add velocity vectors if enabled
        if hasattr(cls, 'actual_vel_arrow') and cls.actual_vel_arrow is not None:
            handles.append(cls.actual_vel_arrow)
            labels.append('Actual Vel')
        
        if hasattr(cls, 'desired_vel_arrow') and cls.desired_vel_arrow is not None:
            handles.append(cls.desired_vel_arrow)
            labels.append('Desired Vel')
        
        # Create the legend
        if handles:
            cls.ax_plots['Map'].legend(handles=handles, labels=labels, loc='upper right')
            
    @classmethod
    def _auto_scale_plot(cls, plot_name, simulation, lines_dict, var_names):
        """Auto-scale a plot based on the data."""
        if not lines_dict:
            return
            
        # Get values
        all_values = []
        for var in var_names:
            if var in cls.history and cls.history[var]:
                all_values.extend(cls.history[var][-50:])  # Last 50 points
        
        if all_values:
            plot_margin = simulation.scenario.config['visualization']['plot_margin']
            y_min = min(all_values) - plot_margin
            y_max = max(all_values) + plot_margin
            cls.ax_plots[plot_name].set_ylim(y_min, y_max)