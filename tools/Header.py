import os
import sys
import logging
import yaml
import csv
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
os.system('cls' if os.name == 'nt' else 'clear')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.simulation.simulator import Simulation
from src.simulation.scenarios import BaseScenario
from src.models.robot import Robot4WSD
from src.utils.config import Load_Config
from src.utils.visualizer import RobotVisualizer as Visualizer
import argparse

# load config
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
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format']
    )
    
    # Create and configure file handler
    file_handler = logging.FileHandler(log_file)
    
    if log_format == 'csv':
        formatter = CSVFormatter(log_fields)
    elif log_format == 'json':
        formatter = JSONFormatter(log_fields)
    else:  # default to text format
        formatter = logging.Formatter(config['logging']['format'])
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
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
        logger.setLevel(log_level)
        # Setup logging
        log_file = setup_logging(config)
        # Initialize simulation
        logger.info("Initializing simulation...")
        sim = Simulation(config)
    except Exception as e:
        logger.error(f"Error setting up simulation: {str(e)}")
    return sim

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
                            f"V=[{action[2]:.1f}, {action[3]:.1f}, {action[4]:.1f}, {action[5]:.1f}]")

                # logger.info(f"T: {sim.time:.1f}, C: δ_f={action[0]:.2f}, δ_r={action[1]:.2f}, "
                #           f"V=[{action[2]:.1f}, {action[3]:.1f}, {action[4]:.1f}, {action[5]:.1f}], "
                #           f"state: {sim.robot.state}, desired: {sim.desired_state}, "
                #           f"global_error: {sim.robot.state - sim.desired_state}")

            if sim.time % 1/sim.save_run_dt < sim.dt:  # Log every second
                # Get the most recently calculated action from input_history instead of recalculating
                action = sim.input_history[sim.time]
                
                # Prepare simulation data for logging
                sim_data = {
                    'time': sim.time,
                    'position': list(sim.robot.state[0:3]),
                    'velocity': list(sim.robot.state[3:6]),
                    'actuator': list(action[0:2]),
                    'wheele_speed': list(action[2:6]),
                }

                # Save simulation data to CSV
                with open(run_file, 'a', newline='') as csvfile:
                    fieldnames = ['time', 'position', 'velocity', 'actuator', 'wheele_speed']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if os.stat(run_file).st_size == 0:  # Write header if file is empty
                        writer.writeheader()
                    writer.writerow(sim_data)
                
            if (sim.time % config['logging']['visualization_frequency'] < sim.dt) and config['visualization']['enabled']:
                Visualizer.plot_results(sim)
                # plt.pause(0.001)

        logger.info("Simulation complete")
        
        # Save final figure
        if config['visualization']['enabled']:
            plt.savefig('simulation_result.png', bbox_inches='tight', dpi=300)
            logger.info("Saved final figure as simulation_result.png")
        
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return 1

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
