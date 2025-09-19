"""Real-time visualization for the robot simulation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Polygon, Circle
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Any
from pathlib import Path
from src.simulation.simulator import Simulation
from src.models.robot import Robot4WSD
import math
import casadi as ca
import logging
import io

try:
    import imageio.v2 as imageio
except ImportError:  # Optional dependency
    imageio = None

logger = logging.getLogger(__name__)


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
            cls.config = simulation.config  # Store config for later use
            # Keep a reference to the simulation object for access in update methods
            cls.simulation = simulation
            cls.enabled_plots = cls._get_enabled_plots(simulation.config)
            
            # Create dynamic layout based on enabled plots
            cls._create_layout(simulation.config)
            
            # For compatibility with existing code
            # Map is now always included in enabled_plots, so this should always work
            cls.ax_robot = cls.ax_plots['Map']
            
            # For other plots, set if available
            cls.ax_states = cls.ax_plots.get('State', None)
            cls.ax_v_omega = cls.ax_plots.get('Velocity', None)
            
            cls.setup_plots(simulation.config)
            # Initialize GIF writer if enabled
            vis_cfg = simulation.config.get('visualization', {})
            cls._gif_frames = None
            cls._video_frames = None
            vis_have_img = imageio is not None
            if vis_cfg.get('save_gif', False) and vis_have_img:
                cls._gif_frames = []
                cls._gif_filename = vis_cfg.get('gif_filename', 'simulation.gif')
                cls._gif_dpi = vis_cfg.get('gif_dpi', 120)
                cls._gif_every = max(1, int(vis_cfg.get('gif_frame_stride', 1)))
                cls._gif_counter = 0
                logger.info(f"GIF capture enabled: file={cls._gif_filename}, stride={cls._gif_every}, dpi={cls._gif_dpi}")
            elif vis_cfg.get('save_gif', False) and not vis_have_img:
                logger.warning("save_gif is True but imageio not installed.")

            # MP4 capture uses same raw frames (store once) then encode
            if vis_cfg.get('save_mp4', False) and vis_have_img:
                cls._video_frames = []
                cls._mp4_filename = vis_cfg.get('mp4_filename', 'simulation.mp4')
                cls._mp4_fps = int(vis_cfg.get('mp4_fps', 15))
                cls._mp4_codec = vis_cfg.get('mp4_codec', 'libx264')
                # Reuse gif stride for capture cadence
                if not hasattr(cls, '_gif_every'):
                    cls._gif_every = 1
                logger.info(f"MP4 capture enabled: file={cls._mp4_filename}, fps={cls._mp4_fps}")
            elif vis_cfg.get('save_mp4', False) and not vis_have_img:
                logger.warning("save_mp4 is True but imageio not installed.")
            # Initialize history with additional fields for Error and Control plots
            cls.history = {
                't': [], 'x': [], 'y': [], 'v': [], 'omega': [],
                'theta': [], 'vx': [], 'vy': [], 'x_d': [], 'y_d': [], 
                'vx_d': [], 'vy_d': [], 'omega_d': [], 'theta_d': [],
                'delta_front': [], 'delta_rear': [], 'V_FL': [], 'V_FR': [], 
                'V_RL': [], 'V_RR': [],
                # CBF constraint values (C_ji = h_dot + alpha * h^2)
                'cbf_distance': [], 'cbf_yielding': [], 'cbf_speed': [], 'cbf_accel': [],
                # Individual barrier function values
                'h_distance': [], 'h_yielding': [], 'h_speed': [], 'h_accel': [],
                # Safety violations (True when C_ji < 0)
                'violation_distance': [], 'violation_yielding': [], 'violation_speed': [], 'violation_accel': []
            }
            cls.robot_patches = []
            cls.path_line = None
            cls.state_lines = {}
            cls.v_omega_lines = {}
            cls.error_lines = {}
            cls.control_lines = {}
            cls.safety_lines = {}
            
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
            
            # Update human visualization if applicable for current scenario
            scenario_name = simulation.scenario.scenario_name
            if ('humans' in simulation.scenario.config['scenario'].get(scenario_name, {})):
                cls._update_human_visualization(simulation)
            
            # Update path line
            if cls.path_line is None:
                cls.path_line, = cls.ax_plots['Map'].plot(cls.history['x'], cls.history['y'], 'b-', alpha=0.5, label='Actual')
            else:
                cls.path_line.set_data(cls.history['x'], cls.history['y'])
            
            # Clear and redraw the desired position marker (red point)
            if hasattr(cls, 'desired_pos_marker') and cls.desired_pos_marker is not None:
                cls.desired_pos_marker.remove()
            cls.desired_pos_marker = cls.ax_plots['Map'].plot(x_d, y_d, 'ro', markersize=cls.config.get('visualization', {}).get('fontsize', 10), label='Desired Pos')[0]
                 # Draw velocity vectors for better visualization (scale them for visibility)
        vel_scale = 0.8  # Scale factor for visibility
        
        # Clear previous velocity vectors
        if hasattr(cls, 'actual_vel_arrow') and cls.actual_vel_arrow is not None:
            cls.actual_vel_arrow.remove()
        if hasattr(cls, 'desired_vel_arrow') and cls.desired_vel_arrow is not None:
            cls.desired_vel_arrow.remove()
            
        # Draw actual velocity vector (blue) - no label to avoid legend issues with arrows
        cls.actual_vel_arrow = cls.ax_plots['Map'].arrow(
            x, y, vx * vel_scale, vy * vel_scale, 
            head_width=0.2, head_length=0.3, fc='blue', ec='blue', 
            alpha=0.7
        )
        
        # Draw desired velocity vector (red) - no label to avoid legend issues with arrows
        cls.desired_vel_arrow = cls.ax_plots['Map'].arrow(
            x_d, y_d, vx_d * vel_scale, vy_d * vel_scale, 
            head_width=0.2, head_length=0.3, fc='red', ec='red', 
            alpha=0.7
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

        # Update fuzzyFunctions plot if enabled
        if 'fuzzyFunctions' in cls.enabled_plots and 'fuzzyFunctions' in cls.ax_plots:
            ax = cls.ax_plots['fuzzyFunctions']
            # Remove previous lines/legend safely (ArtistList has no clear() in some Matplotlib versions)
            try:
                for ln in list(ax.lines):
                    ln.remove()
                # Remove previous collections (e.g., from potential fills) if any
                for coll in list(ax.collections):
                    try: coll.remove()
                    except Exception: pass
                if getattr(ax, 'legend_', None) is not None:
                    ax.legend_.remove()
            except Exception:
                pass
            if hasattr(cls, 'last_lidar'):
                from src.models.fuzzyFunctions import FuzzyFunctions, FuzzyParams
                angles_rad = cls.last_lidar['angles_rad']
                distances = cls.last_lidar['distances']
                max_range = cls.last_lidar['max_range']
                # Goal direction relative to robot heading (deg)
                goal_dir_deg = np.rad2deg(np.arctan2(y_d - y, x_d - x) - theta)
                fz = FuzzyFunctions(FuzzyParams())
                out = fz.compute(angles_rad, distances, max_range, goal_dir_deg=goal_dir_deg)
                deg_grid = out['deg_grid']
                lidar_norm = out['lidar_norm']  # 0..1
                sec_deg = out['sec_deg']
                sec_free_val = out['sec_free_val']
                goal_dir_val = out['goal_dir_val']
                # Scale everything to physical range (0..max_range) instead of normalized
                lidar_dist = lidar_norm * max_range
                # sec_free_scaled = np.array(sec_free_val) * max_range
                # goal_dir_scaled = np.array(goal_dir_val) * max_range
                sec_free_scaled = np.array(sec_free_val)
                goal_dir_scaled = np.array(goal_dir_val)
                ax.plot(deg_grid, lidar_dist, 'r--', linewidth=1.0, label='LIDAR')
                ax.plot(np.r_[-180, sec_deg], np.r_[sec_free_scaled[-1], sec_free_scaled], 'b-', linewidth=2.0, label='Section Free')
                ax.plot(np.r_[-180, sec_deg], np.r_[goal_dir_scaled[-1], goal_dir_scaled], 'k-', linewidth=2.0, label='Goal-Weighted')
                ax.plot([goal_dir_deg, goal_dir_deg], [0, max_range], 'r-', linewidth=1.2, label='Goal Dir')
                ax.set_xlim([-180, 180])
                ax.set_ylim([0, max_range * 1.05])
                ax.set_ylabel(f"Distance / Score (m)")
        
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
            # Evaluate CBF constraint conditions for visualization
            logger.debug(f" SafetyViolation plot is enabled, updating data...")
            cls._update_safety_violation_data(simulation)
            cls._plot_safety_violations()
        else:
            logger.debug(f" SafetyViolation plot not enabled. Enabled plots: {cls.enabled_plots}")
        
        # Update limits for all plots
        window = simulation.scenario.config['visualization']['view_time_window']
        t_min = max(0, simulation.time - window)
        t_max = simulation.time + 0.1
        
        # Set time limits only for time-series plots (exclude Map and fuzzy plots)
        time_series_plots = {'State', 'Velocity', 'Error', 'ControlInput', 'SafetyViolation'}
        for plot_name, ax in cls.ax_plots.items():
            if plot_name in time_series_plots:
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
                    fontsize = cls.config.get('visualization', {}).get('fontsize', 10)
                    setattr(cls, f'{plot_name}_legend', ax.legend(lines, labels, loc='upper right', fontsize=fontsize-1))
            else:
                # Standard legend for other plots
                fontsize = cls.config.get('visualization', {}).get('fontsize', 10)
                ax.legend(loc='upper right', fontsize=fontsize-1)
        
        # Add MPC prediction visualization if controller is MPC
        if 'Map' in cls.enabled_plots and hasattr(simulation, 'controller') and (simulation.config['visualization']['mode'] == 'MPC'):
            cls._update_mpc_visualization(simulation)
        
        cls.fig.canvas.draw()
        cls.fig.canvas.flush_events()

        # Capture GIF frame if enabled
        if getattr(cls, '_gif_frames', None) is not None or getattr(cls, '_video_frames', None) is not None:
            cls._gif_counter += 1
            if cls._gif_counter % cls._gif_every == 0:
                buf = io.BytesIO()
                # Use consistent canvas region (no tight bbox) to keep frame shapes identical
                cls.fig.savefig(buf, format='png', dpi=cls._gif_dpi)
                buf.seek(0)
                frame_img = imageio.imread(buf)
                if getattr(cls, '_gif_frames', None) is not None:
                    cls._gif_frames.append(frame_img)
                if getattr(cls, '_video_frames', None) is not None:
                    cls._video_frames.append(frame_img)
                buf.close()
                if (getattr(cls, '_gif_frames', []) or getattr(cls, '_video_frames', [])) and cls._gif_counter == cls._gif_every:
                    logger.debug("First GIF frame captured.")

    @classmethod
    def finalize_gif(cls):
        """Backward-compatible wrapper to finalize all media."""
        cls.finalize_media()

    @classmethod
    def _normalize_frames(cls, frames):
        if not frames:
            return frames
        try:
            shapes = {fr.shape for fr in frames}
        except Exception:
            return frames
        if len(shapes) == 1:
            return frames
        import numpy as _np
        target = frames[0].shape
        out = []
        ty, tx = target[0], target[1]
        for fr in frames:
            if fr.shape == target:
                out.append(fr)
                continue
            fy, fx = fr.shape[0], fr.shape[1]
            # Crop/pad Y
            if fy > ty:
                sy = (fy - ty)//2
                fr = fr[sy:sy+ty, :]
            elif fy < ty:
                pad_top = (ty - fy)//2
                pad_bottom = ty - fy - pad_top
                fr = _np.pad(fr, ((pad_top,pad_bottom),(0,0),(0,0)), mode='edge')
            # Crop/pad X
            fy, fx = fr.shape[0], fr.shape[1]
            if fx > tx:
                sx = (fx - tx)//2
                fr = fr[:, sx:sx+tx, :]
            elif fx < tx:
                pad_left = (tx - fx)//2
                pad_right = tx - fx - pad_left
                fr = _np.pad(fr, ((0,0),(pad_left,pad_right),(0,0)), mode='edge')
            out.append(fr)
        return out

    @classmethod
    def finalize_media(cls):
        if imageio is None:
            return
        # GIF
        if getattr(cls, '_gif_frames', None) is not None:
            frames = cls._normalize_frames(cls._gif_frames)
            if frames:
                duration = cls.config.get('visualization', {}).get('gif_frame_duration', 0.07)
                try:
                    imageio.mimsave(cls._gif_filename, frames, duration=duration)
                    logger.info(f"Saved GIF animation to {cls._gif_filename} ({len(frames)} frames)")
                except Exception as e:
                    logger.error(f"Failed to save GIF: {e}")
            cls._gif_frames = None
        # MP4
        if getattr(cls, '_video_frames', None) is not None:
            frames_v = cls._normalize_frames(cls._video_frames)
            if frames_v:
                try:
                    writer_args = {}
                    fps = getattr(cls, '_mp4_fps', 15)
                    codec = getattr(cls, '_mp4_codec', 'libx264')
                    # imageio-ffmpeg will pick codec automatically; metadata minimal
                    imageio.mimsave(cls._mp4_filename, frames_v, fps=fps, macro_block_size=None)  # macro_block_size=None avoids resize
                    logger.info(f"Saved MP4 video to {cls._mp4_filename} ({len(frames_v)} frames @ {fps} fps)")
                except Exception as e:
                    logger.error(f"Failed to save MP4: {e}")
            cls._video_frames = None
    
    @classmethod
    def _get_enabled_plots(cls, config: Dict) -> List[str]:
        """Get list of enabled plot types from configuration."""
        enabled_plots = []
        
        # Get plot configuration
        plot_config = config.get('visualization', {}).get('plots', [])
        
        # Parse the configuration and get enabled plots
        for plot_item in plot_config:
            for plot_name, enabled in plot_item.items():
                if enabled:
                    # Map 'position' config to 'Map' internally
                    if plot_name == 'position':
                        enabled_plots.append('Map')
                    else:
                        enabled_plots.append(plot_name)
        
        # Ensure Map is included if not already (default behavior)
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
            
            # Add plots to grid, using polar projection for fuzzyPolar
            for i, plot_name in enumerate(cls.enabled_plots):
                row = i // num_cols
                col = i % num_cols
                if plot_name == 'fuzzyPolar':
                    cls.ax_plots[plot_name] = cls.fig.add_subplot(gs[row, col], projection='polar')
                else:
                    cls.ax_plots[plot_name] = cls.fig.add_subplot(gs[row, col])
    
    @classmethod
    def setup_plots(cls, config):
        """Initialize plot layout."""
        # Get fontsize from config
        fontsize = config.get('visualization', {}).get('fontsize', 10)
        
        # Setup each plot based on its type
        for plot_name, ax in cls.ax_plots.items():
            if plot_name == 'Map':
                # Robot visualization plot
                ax.set_aspect('equal')
                ax.grid(True)
                ax.set_xlabel('x [m]', fontsize=fontsize)
                ax.set_ylabel('y [m]', fontsize=fontsize)
                ax.set_title('Position', fontsize=fontsize)
            elif plot_name == 'State':
                # State plot (velocity states only)
                ax.grid(True)
                ax.set_xlabel('Time [s]', fontsize=fontsize)
                ax.set_ylabel('Velocity States', fontsize=fontsize)
                ax.set_title('Robot Velocity States', fontsize=fontsize)
            elif plot_name == 'Velocity':
                # v/omega plot
                ax.grid(True)
                ax.set_xlabel('Time [s]', fontsize=fontsize)
                ax.set_ylabel('Values', fontsize=fontsize)
                ax.set_title('Speed and Angular Velocity', fontsize=fontsize)
            elif plot_name == 'Error':
                # Error plot
                ax.grid(True)
                ax.set_xlabel('Time [s]', fontsize=fontsize)
                ax.set_ylabel('Error Values', fontsize=fontsize)
                ax.set_title('Tracking Errors', fontsize=fontsize)
            elif plot_name == 'ControlInput':
                # Control input plot
                ax.grid(True)
                ax.set_xlabel('Time [s]', fontsize=fontsize)
                ax.set_ylabel('Steering Angle [rad]', fontsize=fontsize)
                ax.set_title('Control Inputs', fontsize=fontsize)
            elif plot_name == 'SafetyViolation':
                # Safety violation plot
                ax.grid(True)
                ax.set_xlabel('Time [s]', fontsize=fontsize)
                ax.set_ylabel('Violation', fontsize=fontsize)
                ax.set_title('Safety Violations', fontsize=fontsize)
            elif plot_name == 'fuzzyPolar':
                ax.set_title('Obs. and MFs (Polar)')
            elif plot_name == 'fuzzyMF':
                ax.grid(True)
                ax.set_xlabel('Angle [deg]', fontsize=fontsize)
                ax.set_ylabel('Value', fontsize=fontsize)
                ax.set_title('MF and Goal Consideration', fontsize=fontsize)
                ax.set_xlim([-180, 180])
            elif plot_name == 'fuzzyFunctions':
                # Fuzzy functions plot
                ax.grid(True)
                ax.set_xlabel('Angle [deg]', fontsize=fontsize)
                ax.set_ylabel('Membership / Normalized Distance', fontsize=fontsize)
                ax.set_title('Fuzzy LIDAR Section Values', fontsize=fontsize)
                ax.set_xlim([-180, 180])
                ax.set_ylim([0, 1.05])
                
            # Set tick label fontsize for all plots
            ax.tick_params(axis='both', which='major', labelsize=fontsize-1)
        
        # Initialize visualization elements
        cls.human_elements = []
        cls.mpc_elements = []
        
        # Apply tight layout with padding for better margins
        plt.tight_layout(pad=6.0)
    
    @classmethod
    def update_robot(cls, x: float, y: float, theta: float, robot: 'Robot4WSD', 
                    delta_front: float, delta_rear: float, wheel_velocities: np.ndarray):
        """Draw robot with steered wheels and (optionally) LIDAR beams."""
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

        # Draw LIDAR beams if enabled in config
        try:
            show_lidar = False
            lidar_params = {}
            if hasattr(cls, 'config') and 'visualization' in cls.config:
                vis_cfg = cls.config['visualization']
                show_lidar = vis_cfg.get('show_lidar_beams', False)
                lidar_params = vis_cfg.get('lidar_params', {})
            # Also compute beams if any fuzzy plot is enabled
            fuzzy_needed = any(p in getattr(cls, 'enabled_plots', []) for p in ['fuzzyFunctions', 'fuzzyMF', 'fuzzyPolar'])
            compute_beams = show_lidar or fuzzy_needed
            if compute_beams:
                from src.models.Lidar import Lidar
                num_beams = lidar_params.get('num_beams', 32)
                max_range = lidar_params.get('max_range', 3.0)
                fov = lidar_params.get('fov', np.pi*2)
                hit_radius = lidar_params.get('obstacle_radius', None)
                lidar = Lidar(num_beams=num_beams, max_range=max_range, fov=fov)

                # Build obstacle list from scenario humans (circular obstacles)
                obstacles = []
                try:
                    scenario_name = cls.simulation.scenario.scenario_name if hasattr(cls, 'simulation') else None
                except Exception:
                    scenario_name = None
                sim_obj = getattr(cls, 'simulation', None)
                # Fallback: use RobotVisualizer.config and last-known simulation if passed via update
                # We don't have simulation object here directly, so derive from robot and global config if necessary
                # Safest: try to access humans via cls.config if present
                if hasattr(cls, 'current_simulation'):
                    sim_obj = cls.current_simulation

                # Prefer to pull from Visualization context where update_robot is called; attach last sim via class field
                # Construct humans from the latest known scenario config stored in cls.config
                human_positions = []
                try:
                    # When called from plot_results, cls.config is set and simulation object exists.
                    # human_positions expected under config['scenario'][name]['humans']['positions']
                    scenario_cfg = cls.config.get('scenario', {})
                    scenario_name_cfg = scenario_cfg.get('name', None)
                    if scenario_name_cfg and scenario_name_cfg in scenario_cfg:
                        human_positions = scenario_cfg[scenario_name_cfg].get('humans', {}).get('positions', [])
                except Exception:
                    human_positions = []

                # Convert to obstacles with a radius from safety config (rho_0 as conservative human radius)
                default_r = cls.config.get('safety', {}).get('rho_0', 1.5)
                if hit_radius is None:
                    hit_radius = default_r
                for pos in human_positions:
                    if len(pos) >= 2:
                        obstacles.append((float(pos[0]), float(pos[1]), float(hit_radius)))

                # Cast rays
                endpoints, distances, angles = lidar.cast_rays(x, y, theta, obstacles)
                # Store last lidar for other plots
                angles_rel = angles - theta  # relative
                cls.last_lidar = {
                    'angles_rad': angles_rel,
                    'distances': distances,
                    'max_range': max_range,
                    'angles_abs': angles,
                }
                # Plot beams only if explicitly enabled
                if show_lidar:
                    beam_lines = []
                    for end in endpoints:
                        line, = cls.ax_plots['Map'].plot([x, end[0]], [y, end[1]], color='orange', alpha=0.5, linewidth=1)
                        beam_lines.append(line)
                    # Optionally draw hit points
                    hit_pts = cls.ax_plots['Map'].scatter(endpoints[:, 0], endpoints[:, 1], c='orange', s=6, alpha=0.6)
                    beam_lines.append(hit_pts)
                    cls.robot_patches.extend(beam_lines)
        except Exception as e:
            import warnings
            warnings.warn(f"LIDAR visualization failed: {e}")

        # Render fuzzyPolar (subplot 221 equivalent)
        if 'fuzzyPolar' in cls.enabled_plots and 'fuzzyPolar' in cls.ax_plots and hasattr(cls, 'last_lidar'):
            axp = cls.ax_plots['fuzzyPolar']
            # Clear previous artists safely
            try:
                axp.cla()
            except Exception:
                pass
            from src.models.fuzzyFunctions import FuzzyFunctions, FuzzyParams
            # Prepare data
            angles_abs = cls.last_lidar['angles_abs']
            distances = cls.last_lidar['distances']
            max_range = cls.last_lidar['max_range']
            # Compute fuzzy section values for polar overlay rings
            fz = FuzzyFunctions(FuzzyParams())
            # Goal direction from history desired position
            if cls.history.get('x_d') and cls.history['x_d']:
                x_d_last = cls.history['x_d'][-1]
                y_d_last = cls.history['y_d'][-1]
                goal_dir_deg_hist = np.rad2deg(np.arctan2(y_d_last - y, x_d_last - x) - theta)
            else:
                goal_dir_deg_hist = 0.0
            out = fz.compute(cls.last_lidar['angles_rad'], distances, max_range, goal_dir_deg=goal_dir_deg_hist)
            sec_deg = out['sec_deg']
            sec_free_val = out['sec_free_val']
            goal_dir_val = out['goal_dir_val']
            # --- Axis orientation adjustments ---
            # We want robot heading (relative 0 deg) at the TOP (North) instead of right (East).
            # Use polar zero at North so relative fuzzy angles (already centered at 0) appear correctly.
            axp.set_theta_zero_location('N')  # 0 rad up
            axp.set_theta_direction(1)        # Counter-clockwise positive

            # Build unified relative angle grid (include -180 start) then map to [0, 360) for polar continuity
            rel_deg_raw = np.r_[-180, sec_deg]
            # Map to [0,360)
            rel_deg_wrapped = (rel_deg_raw + 360) % 360
            # Sort and keep index mapping for associated values (cyclic start value duplicated at end for closure)
            sort_idx = np.argsort(rel_deg_wrapped)
            rel_deg_sorted = rel_deg_wrapped[sort_idx]
            # Corresponding section values need same ordering; construct arrays aligned with rel_deg_raw
            sec_free_series = np.r_[sec_free_val[-1], sec_free_val]  # align with rel_deg_raw
            goal_dir_series = np.r_[goal_dir_val[-1], goal_dir_val]
            sec_free_sorted = sec_free_series[sort_idx]
            goal_dir_sorted = goal_dir_series[sort_idx]
            # Close loop explicitly
            rel_deg_closed = np.r_[rel_deg_sorted, rel_deg_sorted[0]]
            sec_free_closed = np.r_[sec_free_sorted, sec_free_sorted[0]]
            goal_dir_closed = np.r_[goal_dir_sorted, goal_dir_sorted[0]]
            rel_rad_closed = np.deg2rad(rel_deg_closed)
            axp.plot(rel_rad_closed, sec_free_closed, linewidth=2, color='b', label='Section Free')
            axp.plot(rel_rad_closed, goal_dir_closed, 'k', linewidth=2, label='Goal Weighted')

            # Custom ticks: show -180, -135, -90, -45, 0, 45, 90, 135 (omit duplicate 180) by inverse mapping
            tick_display = np.array([-180, -135, -90, -45, 0, 45, 90, 135])
            tick_positions = np.deg2rad((tick_display + 360) % 360)
            axp.set_xticks(tick_positions)
            axp.set_xticklabels([str(td) for td in tick_display])

            # LIDAR envelope should remain oriented in GLOBAL/world frame (user request: orange one is correct now),
            # but since we rotated axis (zero at North), we COMPENSATE by shifting absolute angles by -90 deg
            # so that global East still appears to the right.
            angles_abs_comp = angles_abs - np.pi/2.0
            lidar_vals = np.clip(distances / max_range, 0, 1)
            # Close global lidar as well
            if len(angles_abs_comp) > 1:
                angles_abs_closed = np.r_[angles_abs_comp, angles_abs_comp[0]]
                lidar_closed = np.r_[lidar_vals, lidar_vals[0]]
            else:
                angles_abs_closed = angles_abs_comp
                lidar_closed = lidar_vals
            axp.plot(angles_abs_closed, lidar_closed, '--', linewidth=2, color='orange', label='LIDAR (norm)')

            # Goal ray (relative): simply from 0 to goal_dir relative angle (convert to rad)
            goal_dir_deg = goal_dir_deg_hist
            goal_dir_rad = np.deg2rad(goal_dir_deg)
            axp.plot([0, goal_dir_rad], [0, min(1, np.mean(distances / max_range))], 'r', linewidth=2, label='Goal Dir')

            axp.set_rlim(0, 1.05)
            axp.set_title('Obs. and MFs (Polar)')
            # Add legend (small font)
            try:
                axp.legend(loc='lower left', fontsize=cls.config.get('visualization', {}).get('fontsize', 8)-1)
            except Exception:
                pass

        # Render fuzzyMF (subplot 222 equivalent)
        if 'fuzzyMF' in cls.enabled_plots and 'fuzzyMF' in cls.ax_plots and hasattr(cls, 'last_lidar'):
            axm = cls.ax_plots['fuzzyMF']
            # Safe clear lines and legend
            try:
                for ln in list(axm.lines):
                    ln.remove()
                # Remove fill_between PolyCollections
                for coll in list(axm.collections):
                    try: coll.remove()
                    except Exception: pass
                if getattr(axm, 'legend_', None) is not None:
                    axm.legend_.remove()
            except Exception:
                pass
            from src.models.fuzzyFunctions import FuzzyFunctions, FuzzyParams
            angles_deg = np.rad2deg(cls.last_lidar['angles_rad'])
            distances = cls.last_lidar['distances']
            max_range = cls.last_lidar['max_range']
            fz = FuzzyFunctions(FuzzyParams())
            goal_dir_deg = goal_dir_deg_hist
            out = fz.compute(cls.last_lidar['angles_rad'], distances, max_range, goal_dir_deg=goal_dir_deg)
            sec_deg = out['sec_deg']
            lidar_norm = out['lidar_norm']
            sec_free_val = out['sec_free_val']
            goal_dir_val = out['goal_dir_val']
            # Plot goal direction and lidar area-like, section and goal-weighted curves
            axm.plot([goal_dir_deg, goal_dir_deg], [0, 1], color='r', linewidth=2, marker='*')
            axm.plot(angles_deg, lidar_norm, 'r--', linewidth=2)
            axm.fill_between(angles_deg, 0, lidar_norm, color='b', alpha=0.15, edgecolor=(0,0,0,0))
            axm.plot(np.r_[-180, sec_deg], np.r_[sec_free_val[-1], sec_free_val], color='b', linewidth=2)
            axm.plot(np.r_[-180, sec_deg], np.r_[goal_dir_val[-1], goal_dir_val], 'k', linewidth=2)
            axm.fill_between(np.r_[-180, sec_deg], 0, np.r_[goal_dir_val[-1], goal_dir_val], color='k', alpha=0.15, edgecolor=(0,0,0,0))
            axm.set_xlim([-180, 180])
            axm.set_ylim([0, 1.05])
    
    @classmethod
    def _update_human_visualization(cls, simulation):
        """Update human visualization elements."""
        scenario_name = simulation.scenario.scenario_name
        scenario_config = simulation.scenario.config['scenario'].get(scenario_name, {})
        human_positions = scenario_config.get('humans', {}).get('positions', [])
        
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
                                          horizontalalignment='center', color='black', fontsize=cls.config.get('visualization', {}).get('fontsize', 10))
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
        scenario_name = simulation.scenario.scenario_name
        scenario_config = simulation.scenario.config['scenario'].get(scenario_name, {})
        if 'humans' in scenario_config:
            for i, pos in enumerate(scenario_config['humans'].get('positions', [])):
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
            logger.debug(f"Warning: Failed to visualize MPC prediction: {e}")
            
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
            logger.debug(f"Warning: Failed to visualize safety barriers: {e}")
            
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
        
        # Add velocity vectors using arrow-like proxy artists for proper legend display
        if hasattr(cls, 'actual_vel_arrow') and cls.actual_vel_arrow is not None:
            # Create proxy line with arrow marker for actual velocity arrow
            actual_vel_proxy = Line2D([0], [0], color='blue', linewidth=2, alpha=0.7, 
                                    marker='>', markersize=10, markerfacecolor='blue', 
                                    markeredgecolor='blue')
            handles.append(actual_vel_proxy)
            labels.append('Actual Vel')
        
        if hasattr(cls, 'desired_vel_arrow') and cls.desired_vel_arrow is not None:
            # Create proxy line with arrow marker for desired velocity arrow
            desired_vel_proxy = Line2D([0], [0], color='red', linewidth=2, alpha=0.7,
                                     marker='>', markersize=10, markerfacecolor='red',
                                     markeredgecolor='red')
            handles.append(desired_vel_proxy)
            labels.append('Desired Vel')
        
        # Add MPC prediction if available
        if hasattr(cls, 'mpc_elements') and cls.mpc_elements:
            for element in cls.mpc_elements:
                if hasattr(element, 'get_label') and element.get_label() == 'MPC Prediction':
                    handles.append(element)
                    labels.append('MPC Prediction')
                    break
        
        # Create the legend
        if handles:
            fontsize = cls.config.get('visualization', {}).get('fontsize', 10)
            cls.ax_plots['Map'].legend(handles=handles, labels=labels, loc='upper right', fontsize=fontsize-1)
            
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

    @classmethod
    def _update_safety_violation_data(cls, simulation):
        """Update safety violation data from the current simulation state.
        
        Evaluates CBF constraints and stores violation information for visualization.
        """
        try:
            # Get current robot state and controller
            state = simulation.robot.state  # Fixed: use simulation.robot.state
            controller = simulation.controller
            
            # Initialize safety violation values to zero
            cbf_values = {
                'cbf_distance': 0.0,
                'cbf_yielding': 0.0, 
                'cbf_speed': 0.0,
                'cbf_accel': 0.0
            }
            h_values = {
                'h_distance': 0.0,
                'h_yielding': 0.0,
                'h_speed': 0.0, 
                'h_accel': 0.0
            }
            violations = {
                'violation_distance': False,
                'violation_yielding': False,
                'violation_speed': False,
                'violation_accel': False
            }
            
            # Extract human states from scenario or controller
            human_states = []
            if hasattr(simulation.scenario, 'config'):
                scenario_name = simulation.scenario.scenario_name
                scenario_config = simulation.scenario.config.get('scenario', {}).get(scenario_name, {})
                if 'humans' in scenario_config:
                    human_positions = scenario_config['humans'].get('positions', [])
                    human_velocities = scenario_config['humans'].get('velocities', [])
                    
                    for pos, vel in zip(human_positions, human_velocities):
                        human_states.append({
                            'x': float(pos[0]),
                            'y': float(pos[1]),
                            'vx': float(vel[0]),
                            'vy': float(vel[1]),
                            'is_goal': False
                        })
            
            # If controller has safety barriers, evaluate them
            logger.debug(f" Controller type: {type(controller).__name__}")
            logger.debug(f" Controller has safety_barriers attribute: {hasattr(controller, 'safety_barriers')}")
            if hasattr(controller, 'safety_barriers'):
                if controller.safety_barriers:
                    barrier_types = [type(barrier).__name__ for barrier in controller.safety_barriers]
                    logger.debug(f" Found {len(controller.safety_barriers)} safety barriers: {barrier_types}")
                else:
                    logger.debug(f" safety_barriers is empty or None")
            
            if hasattr(controller, 'safety_barriers') and controller.safety_barriers:
                logger.debug(f" Processing {len(controller.safety_barriers)} safety barriers")
                robot_state_dict = {
                    'x': float(state[0]),
                    'y': float(state[1]), 
                    'theta': float(state[2]),
                    'vx': float(state[3]),
                    'vy': float(state[4]),
                    'omega': float(state[5])
                }
                logger.debug(f" Robot state: x={robot_state_dict['x']:.2f}, y={robot_state_dict['y']:.2f}, v={np.sqrt(robot_state_dict['vx']**2 + robot_state_dict['vy']**2):.2f}")
                logger.debug(f" Found {len(human_states)} humans")
                
                # Get current control input if available
                robot_input_dict = {}
                if hasattr(controller, 'last_u') and controller.last_u is not None:
                    robot_input_dict = {
                        'delta_front': float(controller.last_u[0]),
                        'delta_rear': float(controller.last_u[1]),
                        'V_FL': float(controller.last_u[2]),
                        'V_FR': float(controller.last_u[3]),
                        'V_RL': float(controller.last_u[4]),
                        'V_RR': float(controller.last_u[5])
                    }
                
                # Evaluate each safety barrier
                barrier_names = ['distance', 'yielding', 'speed', 'accel']
                for i, barrier in enumerate(controller.safety_barriers):
                    if i < len(barrier_names):
                        barrier_name = barrier_names[i]
                        logger.debug(f" Evaluating {barrier_name} barrier")
                        
                        try:
                            # Set robot state in barrier
                            barrier.set_robot_state(robot_state_dict)
                            if robot_input_dict:
                                barrier.set_robot_input(robot_input_dict)
                            
                            # For distance and yielding barriers, evaluate against humans
                            if barrier_name in ['distance', 'yielding'] and human_states:
                                min_h = float('inf')
                                min_cbf = float('inf')
                                max_violation = False
                                
                                for human in human_states:
                                    if not human.get('is_goal', False):  # Skip goal position
                                        try:
                                            h_val = barrier.h(human)
                                            h_dot_val = barrier.h_dot(human) 
                                            alpha = barrier.get_adaptive_alpha(robot_state_dict)
                                            
                                            # CBF condition: C_ji = h_dot + alpha * h^2
                                            cbf_condition = h_dot_val + alpha * (h_val ** 2)
                                            
                                            logger.debug(f" {barrier_name} barrier - h={h_val:.3f}, h_dot={h_dot_val:.3f}, alpha={alpha:.3f}, CBF={cbf_condition:.3f}")
                                            
                                            min_h = min(min_h, h_val)
                                            min_cbf = min(min_cbf, cbf_condition)
                                            
                                            # Violation occurs when CBF condition < 0
                                            if cbf_condition < 0:
                                                max_violation = True
                                                logger.debug(f" VIOLATION DETECTED for {barrier_name}!")
                                                
                                        except Exception as e:
                                            logger.debug(f" Error evaluating {barrier_name} barrier: {e}")
                                            # Handle barrier evaluation errors gracefully
                                            continue
                                
                                if min_h != float('inf'):
                                    h_values[f'h_{barrier_name}'] = min_h
                                    cbf_values[f'cbf_{barrier_name}'] = min_cbf
                                    violations[f'violation_{barrier_name}'] = max_violation
                                    
                            # For speed and accel barriers, evaluate directly
                            elif barrier_name in ['speed', 'accel']:
                                try:
                                    # Use empty human state for speed/accel barriers (they don't depend on humans)
                                    dummy_human = {'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 0.0}
                                    h_val = barrier.h(dummy_human)
                                    h_dot_val = barrier.h_dot(dummy_human)
                                    alpha = barrier.get_adaptive_alpha(robot_state_dict)
                                    
                                    # CBF condition: C_ji = h_dot + alpha * h^2
                                    cbf_condition = h_dot_val + alpha * (h_val ** 2)
                                    
                                    h_values[f'h_{barrier_name}'] = h_val
                                    cbf_values[f'cbf_{barrier_name}'] = cbf_condition
                                    violations[f'violation_{barrier_name}'] = cbf_condition < 0
                                    
                                except Exception as e:
                                    # Handle barrier evaluation errors gracefully
                                    logger.debug(f" Barrier evaluation error for {barrier_name}: {e}")
                                    continue
                                    
                        except Exception as e:
                            # Handle barrier setup errors gracefully
                            logger.debug(f" Barrier setup error for {barrier_name}: {e}")
                            continue
            
            # Store values in history
            for key, value in cbf_values.items():
                cls.history[key].append(value)
            for key, value in h_values.items():
                cls.history[key].append(value)
            for key, value in violations.items():
                cls.history[key].append(value)
                
        except Exception as e:
            # Fallback: append zeros if there's any error
            safety_keys = [
                'cbf_distance', 'cbf_yielding', 'cbf_speed', 'cbf_accel',
                'h_distance', 'h_yielding', 'h_speed', 'h_accel',
                'violation_distance', 'violation_yielding', 'violation_speed', 'violation_accel'
            ]
            for key in safety_keys:
                if key.startswith('violation_'):
                    cls.history[key].append(False)
                else:
                    cls.history[key].append(0.0)

    @classmethod 
    def _plot_safety_violations(cls):
        """Plot safety violation data on the SafetyViolation subplot."""
        try:
            if not cls.history['t']:
                return
                
            ax = cls.ax_plots['SafetyViolation']
            
            # Initialize safety lines dictionary if it doesn't exist
            if not hasattr(cls, 'safety_lines'):
                cls.safety_lines = {}
            
            # Define CBF constraint variables to plot
            cbf_vars = {
                'cbf_distance': ('r-', 'Distance CBF (C_d)'),
                'cbf_yielding': ('b-', 'Yielding CBF (C_y)'), 
                'cbf_speed': ('g-', 'Speed CBF (C_s)'),
                'cbf_accel': ('m-', 'Accel CBF (C_a)')
            }
            
            # Define barrier function variables to plot (with different line style)
            h_vars = {
                'h_distance': ('r--', 'Distance h(x)'),
                'h_yielding': ('b--', 'Yielding h(x)'),
                'h_speed': ('g--', 'Speed h(x)'),
                'h_accel': ('m--', 'Accel h(x)')
            }
            
            # Initialize violation legend proxies if not already done
            if not hasattr(cls, 'violation_proxies_initialized'):
                cls.violation_proxies_initialized = True
                
                # Create proxy patches for violation types that will be shown in legend
                violation_types = ['Distance', 'Yielding', 'Speed', 'Accel']
                violation_colors = ['red', 'blue', 'green', 'magenta']
                
                for vtype, color in zip(violation_types, violation_colors):
                    proxy_key = f'violation_proxy_{vtype.lower()}'
                    # Create an invisible line as proxy for violation shading
                    cls.safety_lines[proxy_key], = ax.plot([], [], color=color, alpha=0.3, 
                                                          linewidth=8, label=f'{vtype} Violation')
            
            # Plot CBF constraint values (primary indicators)
            for var, (style, label) in cbf_vars.items():
                if var not in cls.safety_lines:
                    cls.safety_lines[var], = ax.plot(cls.history['t'], cls.history[var], 
                                                   style, label=label, linewidth=2)
                else:
                    cls.safety_lines[var].set_data(cls.history['t'], cls.history[var])
            
            # Plot barrier function values (secondary indicators)
            for var, (style, label) in h_vars.items():
                if var not in cls.safety_lines:
                    cls.safety_lines[var], = ax.plot(cls.history['t'], cls.history[var], 
                                                   style, label=label, alpha=0.7)
                else:
                    cls.safety_lines[var].set_data(cls.history['t'], cls.history[var])
            
            # Plot safety threshold line at y=0 (CBF condition: C_ji >= 0)
            if 'safety_threshold' not in cls.safety_lines:
                cls.safety_lines['safety_threshold'], = ax.plot(cls.history['t'], 
                                                              [0] * len(cls.history['t']), 
                                                              'k--', label='Safety Threshold', 
                                                              linewidth=2, alpha=0.8)
            else:
                cls.safety_lines['safety_threshold'].set_data(cls.history['t'], 
                                                            [0] * len(cls.history['t']))
            
            # Clear previous violation shading (but not the proxy lines)
            if hasattr(cls, 'violation_patches'):
                for patch in cls.violation_patches:
                    try:
                        patch.remove()
                    except:
                        pass
                cls.violation_patches = []
            else:
                cls.violation_patches = []
            
            # Highlight violation regions without adding new legend entries
            violation_vars = ['violation_distance', 'violation_yielding', 'violation_speed', 'violation_accel']
            violation_colors = ['red', 'blue', 'green', 'magenta']
            
            for i, (var, color) in enumerate(zip(violation_vars, violation_colors)):
                if cls.history[var]:
                    # Find violation periods and shade them
                    violations = np.array(cls.history[var])
                    times = np.array(cls.history['t'])
                    
                    # Create violation spans
                    violation_indices = np.where(violations)[0]
                    if len(violation_indices) > 0:
                        # Group consecutive violation indices
                        violation_spans = []
                        start_idx = violation_indices[0]
                        end_idx = start_idx
                        
                        for idx in violation_indices[1:]:
                            if idx == end_idx + 1:
                                end_idx = idx
                            else:
                                violation_spans.append((start_idx, end_idx))
                                start_idx = idx
                                end_idx = idx
                        violation_spans.append((start_idx, end_idx))
                        
                        # Plot violation spans without labels (legend already exists from proxies)
                        for span_start, span_end in violation_spans:
                            if span_start < len(times) and span_end < len(times):
                                patch = ax.axvspan(times[span_start], times[span_end], 
                                                 alpha=0.2, color=color)
                                cls.violation_patches.append(patch)
            
            # Set y-axis to show both positive and negative values
            if len(cls.history['t']) > 1:
                all_cbf_values = []
                all_h_values = []
                
                for var in cbf_vars.keys():
                    if cls.history[var]:
                        all_cbf_values.extend(cls.history[var])
                        
                for var in h_vars.keys():
                    if cls.history[var]:
                        all_h_values.extend(cls.history[var])
                
                if all_cbf_values or all_h_values:
                    all_values = all_cbf_values + all_h_values
                    y_min = min(all_values + [-0.5])  # Include some negative space
                    y_max = max(all_values + [1.0])   # Include some positive space
                    
                    # Add margin
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)
            
            # Update legend only once with predefined entries
            if not hasattr(cls, 'safety_legend_created'):
                cls.safety_legend_created = True
                    
        except Exception as e:
            # Handle plotting errors gracefully
            logger.debug(f"Warning: SafetyViolation plot update failed: {e}")
            pass

    @classmethod
    def save_individual_subplots(cls, output_dir: str = "plots"):
        """
        Save each enabled subplot as a separate PNG file to the specified directory.
        Uses matplotlib's built-in functionality to extract subplots cleanly.
        
        Args:
            output_dir: Directory to save individual subplot files (default: "plots")
        """
        import os
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not hasattr(cls, 'ax_plots') or not cls.ax_plots:
            logger.warning("No plots available to save individually")
            return
            
        logger.info(f"Saving individual subplots to {output_dir}/")
        
        # Save each enabled subplot individually using matplotlib's extent functionality
        for plot_name in cls.enabled_plots:
            if plot_name in cls.ax_plots and plot_name != 'dummy':
                try:
                    # Get the subplot axis
                    ax = cls.ax_plots[plot_name]
                    
                    # Map 'Map' back to 'position' for filename
                    filename_plot_name = 'position' if plot_name == 'Map' else plot_name.lower()
                    filename = f"subplot_{filename_plot_name}.png"
                    filepath = output_path / filename
                    
                    # Save just this subplot using its extent with better margins
                    extent = ax.get_window_extent().transformed(cls.fig.dpi_scale_trans.inverted())
                    cls.fig.savefig(filepath, bbox_inches=extent.expanded(1.35, 1.25), dpi=600)
                    
                    display_name = 'Position' if plot_name == 'Map' else plot_name
                    logger.info(f"Saved {display_name} subplot as {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to save {plot_name} subplot: {e}")


class MPCVisualizer:
    """Visualization methods for MPC controller"""

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
                
                # Get the robot model for prediction
                if hasattr(controller, 'robot_model'):
                    robot_model = controller.robot_model
                else:
                    # Fallback: create a temporary robot model if not available
                    from src.models.robot import Robot4WSD
                    robot_model = Robot4WSD(controller.config)
                
                # Get time step
                dt = controller.params.dt if hasattr(controller, 'params') else 0.01
                
                # Simulate forward using robot's predict method
                state = current_state.copy()
                
                # For each step in control horizon
                for k in range(min(Hp, len(controller.last_solution))):
                    # Extract control action from the cached solution
                    u = controller.last_solution[k]
                    
                    # Use robot's predict method for accurate forward simulation
                    state = robot_model.predict(state, u, dt)
                    predicted_states.append(state)
                
                return predicted_states
            return None
        except Exception as e:
            logger.debug(f"Warning: Could not extract MPC prediction: {e}")
            return None