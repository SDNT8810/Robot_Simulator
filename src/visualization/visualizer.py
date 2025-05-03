"""Real-time visualization for the robot simulation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Polygon
from typing import List, Tuple
from src.simulation.simulator import Simulation
from src.models.robot import Robot4WSD

class RobotVisualizer:
    """Real-time visualization of the robot state."""
    
    @classmethod
    def plot_results(cls, simulation: 'Simulation'):
        """Plot simulation results."""
        # Get current state and inputs
        state = simulation.state
        x, y, theta = state[0:3]
        vx, vy, omega = state[3:6]

        # Get current control inputs
        current_input = simulation.get_control_inputs().get(simulation.time, np.zeros(6))
        delta_front, delta_rear = current_input[0:2]
        wheel_velocities = current_input[2:6]
        
        # Create figure if it doesn't exist
        if not hasattr(cls, 'fig'):
            cls.fig = plt.figure(figsize=(16, 12))  # Increased height for 2x2 layout
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
                        
            gs = cls.fig.add_gridspec(2, 2)  # Changed to 2x2 grid
            cls.ax_robot = cls.fig.add_subplot(gs[:, 0])  # Robot visualization takes left half
            cls.ax_states = cls.fig.add_subplot(gs[0, 1])  # States plot top right
            cls.ax_v_omega = cls.fig.add_subplot(gs[1, 1])  # v/omega plot bottom right
            
            cls.setup_plots()
            cls.history = {
                't': [], 'x': [], 'y': [], 'v': [], 'omega': [],
                'theta': [], 'vx': [], 'vy': []
            }
            cls.robot_patches = []
            cls.path_line = None
            cls.state_lines = {}
            cls.v_omega_lines = {}
            
            # Plot start and goal positions
            start_pos = simulation.scenario.get_initial_state()[:2]
            if simulation.scenario.senario_name == 'to_goal':
                goal_pos = simulation.scenario.goal[:2]
            else:  # circle scenario
                goal_pos = simulation.scenario.center
                
            cls.ax_robot.plot(start_pos[0], start_pos[1], 'g*', markersize=15, label='Start', zorder=5)
            cls.ax_robot.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Target', zorder=5)
            
            # Create desired circle path if applicable
            if simulation.scenario.senario_name == 'circle':
                radius = simulation.scenario.radius
                center = simulation.scenario.center
                t = np.linspace(0, 2*np.pi, 100)
                x_circle = center[0] + radius * np.cos(t)
                y_circle = center[1] + radius * np.sin(t)
                cls.ax_robot.plot(x_circle, y_circle, 'g-.', label='Desired', linewidth=2)
                
        # Store history
        cls.history['t'].append(simulation.time)
        cls.history['x'].append(x)
        cls.history['y'].append(y)
        cls.history['theta'].append(theta)
        cls.history['vx'].append(vx)
        cls.history['vy'].append(vy)
        cls.history['omega'].append(omega)
        
        # Calculate actual speed
        v = np.sqrt(vx**2 + vy**2)
        cls.history['v'].append(v)

        # Clear previous robot visualization
        for patch in cls.robot_patches:
            patch.remove()
        cls.robot_patches.clear()
        
        # Update visualization with steering angles
        cls.update_robot(x, y, theta, simulation.robot, delta_front, delta_rear, wheel_velocities)
        
        # Update path line
        if cls.path_line is None:
            cls.path_line, = cls.ax_robot.plot(cls.history['x'], cls.history['y'], 'b-', alpha=0.5, label='Actual')
        else:
            cls.path_line.set_data(cls.history['x'], cls.history['y'])
            
        # Update state plots (only velocity states)
        state_vars = {
            'vx': ['m-', 'Vx [m/s]'],
            'vy': ['c-', 'Vy [m/s]'],
            'omega': ['y-', 'ω [rad/s]']
        }
        
        for var, (style, label) in state_vars.items():
            if var not in cls.state_lines:
                cls.state_lines[var], = cls.ax_states.plot(cls.history['t'], 
                                                         cls.history[var], 
                                                         style, 
                                                         label=label)
            else:
                cls.state_lines[var].set_data(cls.history['t'], cls.history[var])
        
        # Update v/omega plot
        if 'v' not in cls.v_omega_lines:
            cls.v_omega_lines['v'], = cls.ax_v_omega.plot(cls.history['t'], 
                                                         cls.history['v'], 
                                                         'g-', 
                                                         label='v [m/s]')
            cls.v_omega_lines['omega'], = cls.ax_v_omega.plot(cls.history['t'], 
                                                             cls.history['omega'], 
                                                             'r-', 
                                                             label='ω [rad/s]')
        else:
            cls.v_omega_lines['v'].set_data(cls.history['t'], cls.history['v'])
            cls.v_omega_lines['omega'].set_data(cls.history['t'], cls.history['omega'])
        
        # Update limits and create legends only after data is plotted
        margin = simulation.scenario.config['visualization']['margins']['margin']
        if simulation.scenario.config['visualization']['margins']['mode'] == 'Auto_Center':
            cls.ax_robot.set_xlim(x - margin, x + margin)
            cls.ax_robot.set_ylim(y - margin, y + margin)
        else:
            x_center, y_center = simulation.config['scenario']['circle']['center']
            cls.ax_robot.set_xlim(x_center - margin, x_center + margin)
            cls.ax_robot.set_ylim(y_center - margin, y_center + margin)
        
        window = simulation.scenario.config['visualization']['view_time_window']
        t_min = max(0, simulation.time - window)
        t_max = simulation.time + 0.1
        
        cls.ax_states.set_xlim(t_min, t_max)
        cls.ax_v_omega.set_xlim(t_min, t_max)
        
        if cls.history['t']:  # Only create legends if we have data
            # Auto-scale state plot (velocity states only)
            state_margin = simulation.scenario.config['visualization']['margins']['plot']
            all_states = []
            for var in state_vars:
                all_states.extend(cls.history[var][-50:])  # Last 50 points
            if all_states:
                y_min = min(all_states) - state_margin
                y_max = max(all_states) + state_margin
                cls.ax_states.set_ylim(y_min, y_max)
                if len(cls.state_lines) > 0:  # Only create legend if we have plotted lines
                    cls.ax_states.legend(loc='upper right')
            
            # Auto-scale v/omega plot
            v_omega_margin = simulation.scenario.config['visualization']['margins']['plot']
            v_omega_data = cls.history['v'][-50:] + cls.history['omega'][-50:]
            if v_omega_data:
                y_min = min(v_omega_data) - v_omega_margin
                y_max = max(v_omega_data) + v_omega_margin
                cls.ax_v_omega.set_ylim(y_min, y_max)
                if len(cls.v_omega_lines) > 0:  # Only create legend if we have plotted lines
                    cls.ax_v_omega.legend(loc='upper right')
        
        # Robot plot legend is created last and only if we have the path line
        if cls.path_line is not None:
            cls.ax_robot.legend(loc='upper right')
        
        cls.fig.canvas.draw()
        cls.fig.canvas.flush_events()
    
    @classmethod
    def setup_plots(cls):
        """Initialize plot layout."""
        # Robot visualization plot
        cls.ax_robot.set_aspect('equal')
        cls.ax_robot.grid(True)
        cls.ax_robot.set_xlabel('x [m]')
        cls.ax_robot.set_ylabel('y [m]')
        
        # State plot (velocity states only)
        cls.ax_states.grid(True)
        cls.ax_states.set_xlabel('Time [s]')
        cls.ax_states.set_ylabel('Velocity States')
        cls.ax_states.set_title('Robot Velocity States')
        
        # v/omega plot
        cls.ax_v_omega.grid(True)
        cls.ax_v_omega.set_xlabel('Time [s]')
        cls.ax_v_omega.set_ylabel('Values')
        cls.ax_v_omega.set_title('Speed and Angular Velocity')
        
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
        line, = cls.ax_robot.plot(points[:, 0], points[:, 1], 'k-', linewidth=2)
        cls.robot_patches.append(line)
        
        # Draw direction arrow
        arrow = Arrow(x, y, length/2*c, length/2*s, width=0.3, color='red')
        cls.ax_robot.add_patch(arrow)
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
            cls.ax_robot.add_patch(wheel)
            cls.robot_patches.append(wheel)
            
            # Add direction indicator on wheel
            indicator_length = wheel_length * 0.8
            indicator_start = wheel_pos
            indicator_end = wheel_pos + indicator_length * np.array([c_wheel, s_wheel])
            indicator, = cls.ax_robot.plot([indicator_start[0], indicator_end[0]], 
                                         [indicator_start[1], indicator_end[1]], 
                                         'k-', linewidth=1)
            cls.robot_patches.append(indicator)