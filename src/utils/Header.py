import os
import sys
import logging
import yaml
import csv
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
os.system('cls' if os.name == 'nt' else 'clear')

# Configure logging - will be overridden by setup_logging() from config
logger = logging.getLogger(__name__)

from src.simulation.simulator import Simulation
from src.simulation.scenarios import BaseScenario
from src.models.robot import Robot4WSD
from src.utils.config import Load_Config
from src.utils.visualizer import RobotVisualizer as Visualizer
from src.safety.barrier import extract_safety_data
import argparse

def cleanup_logs_and_runs(project_root=None, keep_logs: int = 5, keep_runs: int = 10) -> None:
    """
    Clean up old log files and run data to free up space.
    
    Args:
        project_root: Path to the project root directory. If None, uses the current project root.
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    print("🧹 Starting cleanup of logs and runs...")
    
    # Clean up logs directory
    logs_dir = project_root / 'logs'
    if logs_dir.exists():
        # remove all files inside logs directory
        for log_file in logs_dir.glob('*.txt'):
            log_file.unlink()
            print(f"📄 Deleted log file: {log_file.name}")
    else:
        print("📁 Logs directory doesn't exist")
    
    # Clean up runs directory
    runs_dir = project_root / 'runs'
    if runs_dir.exists():
        # remove all files inside runs directory
        for run_file in runs_dir.glob('*.csv'):
            run_file.unlink()
            print(f"📊 Deleted run file: {run_file.name}")
    else:
        print("📁 Runs directory doesn't exist")
    
    # Clean up plots directory
    plots_dir = project_root / 'plots'
    if plots_dir.exists():
        # remove all files inside plots directory
        for plot_file in plots_dir.glob('*.png'):
            plot_file.unlink()
            print(f"📊 Deleted plot file: {plot_file.name}")
    else:
        print("📁 Plots directory doesn't exist")

    # # Clean up pyc files and __pycache__ directories in src directory
    # src_dir = project_root / 'src'
    # if src_dir.exists():
    #     # remove all pyc files inside src directory
    #     for pyc_file in src_dir.glob('**/*.pyc'):
    #         pyc_file.unlink()
    #         print(f"�️ Deleted pyc file: {pyc_file.name}")
    #     # remove all __pycache__ directories inside src directory
    #     for pycache_dir in src_dir.glob('**/__pycache__'):
    #         shutil.rmtree(pycache_dir)
    #         print(f"🗑️ Deleted __pycache__ directory: {pycache_dir.name}")
    # else:
    #     print("📁 Src directory doesn't exist")

    print("🎯 Cleanup complete!\n")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def override_config(config: dict, args) -> dict:
    """Override configuration with command line arguments."""
    if args.param:
        logger.info("Overriding parameters:")
        for param, value in args.param:
            logger.info(f"  {param} = {value}")
            # Try to convert string to number if possible
            try:
                value = float(value)
            except ValueError:
                pass
            # Update nested config using parameter path
            current = config
            keys = param.split('.')
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value

    return config

def deep_update_dict(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = deep_update_dict(base[k], v)
        else:
            base[k] = v
    return base

def setup_logging(config):
    """Setup logging configuration based on config parameters."""
    log_path = Path(config['logging']['log_path'])
    log_path.mkdir(exist_ok=True)
    
    log_file = log_path / config['logging']['log_file']
    log_format = config['logging']['log_format'].lower()
    log_fields = config['logging']['log_data']
    
    # Get logging level from config
    log_level = getattr(logging, config['logging']['level'])
    
    # Configure root logger with force=True to override existing configuration
    logging.basicConfig(
        level=log_level,
        format=config['logging']['format'],
        force=True
    )
    
    # Suppress verbose matplotlib font DEBUG messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
    
    # Update all existing loggers to inherit the new level
    # This ensures that module-level loggers respect the config level
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Update all existing loggers that don't have explicit levels set
    for name in logging.Logger.manager.loggerDict:
        existing_logger = logging.getLogger(name)
        if existing_logger.level == logging.NOTSET:  # Only update loggers with no explicit level
            existing_logger.setLevel(log_level)
    
    # Create and configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)  # Set the file handler level to match config
    
    if log_format == 'csv':
        formatter = CSVFormatter(log_fields)
    elif log_format == 'json':
        formatter = JSONFormatter(log_fields)
    else:  # default to text format
        formatter = logging.Formatter(config['logging']['format'])
    
    file_handler.setFormatter(formatter)
    
    # Add file handler to root logger so all loggers write to file
    root_logger.addHandler(file_handler)
    
    return log_file

def setup_simulation(args) -> 'Simulation':
    """Setup simulation environment based on configuration."""
    sim = None
    try:
        # Always load default config first
        default_config_path = project_root / 'config' / 'default_config.yaml'
        if not default_config_path.exists():
            logger.error(f"Default config file not found: {default_config_path}")
            return None
        logger.info(f"Loading default config from {default_config_path}")
        config = Load_Config(default_config_path)

        # If temporary_config.yaml exists, override defaults
        temp_config_path = project_root / 'config' / 'temporary_config.yaml'
        if temp_config_path.exists():
            logger.info(f"Overriding with temporary config: {temp_config_path}")
            temp_config = Load_Config(temp_config_path)
            config = deep_update_dict(config, temp_config)

        # If --config is specified and not empty, not default_config.yaml or temporary_config.yaml, override again
        if hasattr(args, 'config') and args.config and args.config not in ['default_config.yaml', 'temporary_config.yaml']:
            config_arg_path = project_root / 'config' / args.config
            if config_arg_path.exists():
                logger.info(f"Overriding with config: {config_arg_path}")
                config_arg = Load_Config(config_arg_path)
                config = deep_update_dict(config, config_arg)

        # Override with command line parameters if provided
        config = override_config(config, args)
        log_level = config.get('logging', {}).get('level', 'INFO')
        logger.setLevel(getattr(logging, log_level))
        # Setup logging
        log_file = setup_logging(config)
        # Initialize simulation
        logger.info("Initializing simulation...")
        sim = Simulation(config)
    except Exception as e:
        logger.error(f"Error setting up simulation: {str(e)}")
    return sim

def _get_dynamic_csv_fieldnames(num_humans=0):
    """Generate dynamic CSV fieldnames based on number of humans.
    
    Args:
        num_humans: Number of humans in the simulation
        
    Returns:
        List of fieldnames for CSV logging with flat structure for analysis
    """
    # Base fieldnames with flat structure for easier analysis
    base_fieldnames = [
        'time',
        # Position components (3D)
        'pos_x', 'pos_y', 'pos_z',
        # Velocity components (3D) 
        'vel_x', 'vel_y', 'vel_z',
        # Actuator inputs
        'actuator_front', 'actuator_rear',
        # Wheel speeds
        'wheel_speed_fl', 'wheel_speed_fr', 'wheel_speed_rl', 'wheel_speed_rr'
    ]
    
    # Aggregated safety fieldnames (backward compatibility)
    safety_fieldnames = [
        'num_humans',  # Add number of humans for reference
        'h_distance', 'h_yielding', 'h_speed', 'h_accel',
        'h_dot_distance', 'h_dot_yielding', 'h_dot_speed', 'h_dot_accel',
        'cbf_distance', 'cbf_yielding', 'cbf_speed', 'cbf_accel',
        'violation_distance', 'violation_yielding', 'violation_speed', 'violation_accel',
        'alpha_distance', 'alpha_yielding', 'alpha_speed', 'alpha_accel'
    ]
    
    # Per-human safety fieldnames (dynamic based on number of humans)
    per_human_fieldnames = []
    barrier_types = ['distance', 'yielding', 'speed', 'accel']  # All 4 barriers for each human
    metrics = ['h', 'h_dot', 'cbf', 'violation', 'alpha']
    
    for human_idx in range(num_humans):
        for barrier_type in barrier_types:
            for metric in metrics:
                field_name = f'{metric}_{barrier_type}_human{human_idx}'
                per_human_fieldnames.append(field_name)
    
    return base_fieldnames + safety_fieldnames + per_human_fieldnames

def run_simulation(sim: Simulation) -> int:
    """Run the robot simulation with given arguments.
    
    Args:
        sim: Simulation object
        
    Returns:
        0 on success, 1 on error
    """
    try:
        # Run simulation loop
        logger.info("Starting simulation loop...")
        config = sim.config

        run_dir = Path('runs')
        run_dir.mkdir(exist_ok=True)
        
        # Create run-specific directory and file
        run_name = config['logging']['run_name']
        run_file = run_dir / f"run_{run_name}.csv"
                
        while sim.step():
            # Log at regular intervals based on log_dt
            if sim.time % sim.log_dt < sim.dt:
                # Get the most recently calculated action from input_history instead of recalculating
                action = sim.input_history[sim.time]
                
                # # Log simulation data to console
                logger.info(f"T: {sim.time:.1f}, C: δ_f={action[0]:.2f}, δ_r={action[1]:.2f}, "
                          f"V=[{action[2]:.1f}, {action[3]:.1f}, {action[4]:.1f}, {action[5]:.1f}], "
                          f"state: {sim.robot.state}, desired: {sim.desired_state}, "
                          f"global_error: {sim.robot.state - sim.desired_state}")

            if sim.time % 1/sim.save_run_dt < sim.dt:  # Log every second
                # Generate dynamic fieldnames based on number of humans detected
                num_humans = sim.safety_data.get('num_humans', 0)
                fieldnames = _get_dynamic_csv_fieldnames(num_humans)

                # Save simulation data to CSV
                with open(run_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if os.stat(run_file).st_size == 0:  # Write header if file is empty
                        writer.writeheader()
                    writer.writerow(sim.sim_data)
                
            if (sim.time % config['logging']['visualization_frequency'] < sim.dt) and config['visualization']['enabled']:
                Visualizer.plot_results(sim)
                plt.pause(0.001)

        logger.info("Simulation complete")

        # Save final figure
        if config['visualization']['enabled']:
            plt.savefig('simulation_result.png', bbox_inches='tight', dpi=600)
            logger.info("Saved final figure as simulation_result.png")

            # Save individual subplots to plots directory
            Visualizer.save_individual_subplots()

        return 0

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return 1
    finally:
        # Always attempt to finalize GIF (safe no-op if disabled)
        try:
            if hasattr(Visualizer, 'finalize_gif'):
                Visualizer.finalize_gif()
        except Exception as _gif_err:
            logger.warning(f"Failed to finalize GIF: {_gif_err}")
class CSVFormatter(logging.Formatter):
    """Custom formatter for CSV logs"""
    def __init__(self, fields):
        super().__init__()
        self.fields = fields
    
    def format(self, record):
        # Create a dictionary of the record's attributes
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage()
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'sim_data'):
            for field in self.fields:
                if field in record.sim_data:
                    log_data[field] = record.sim_data[field]
        
        # Convert to CSV line
        if not hasattr(self, 'writer'):
            self.writer = csv.DictWriter(open(record.logfile, 'a'), fieldnames=log_data.keys())
            if os.stat(record.logfile).st_size == 0:  # Write header if file is empty
                self.writer.writeheader()
        
        return self.writer.writerow(log_data)
class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON logs"""
    def __init__(self, fields):
        super().__init__()
        self.fields = fields
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage()
        }
        
        if hasattr(record, 'sim_data'):
            for field in self.fields:
                if field in record.sim_data:
                    log_data[field] = record.sim_data[field]
        
        return json.dumps(log_data)
