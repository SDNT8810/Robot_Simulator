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
from src.visualization.visualizer import RobotVisualizer as Visualizer
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

def run_simulation(args) -> int:
    """Run the robot simulation with given arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        0 on success, 1 on error
    """
    try:
        # Load base config
        config_path = project_root / 'config' / args.config
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
            
        logger.info(f"Loading config from {config_path}")
        config = Load_Config(config_path)
        
        # Override with command line parameters if provided
        config = override_config(config, args)
        log_level = config.get('logging', {}).get('level', 'INFO')
        logger.setLevel(log_level)
        
        # Setup logging
        log_file = setup_logging(config)

        # Initialize simulation
        logger.info("Initializing simulation...")
        scenario = BaseScenario(config)
        sim = Simulation(scenario)
        
        # Run simulation loop
        logger.info("Starting simulation loop...")
        while sim.step():
            if sim.time % sim.log_dt < sim.dt:  # Log every second
                action = sim.controller.action(sim.state, sim.desired_state)
                
                # Prepare simulation data for logging
                sim_data = {
                    'time': sim.time,
                    'position': list(sim.state[0:3]),
                    'velocity': list(sim.state[3:6]),
                    'actuator': list(sim.action),
                    'control_input': list(action),
                    'safety_violation': False  # Add safety violation status if available
                }
                
                # Log simulation data
                logger.info(f"T: {sim.time:.1f}, C: δ_f={action[0]:.2f}, δ_r={action[1]:.2f}, "
                          f"V=[{action[2]:.1f}, {action[3]:.1f}, {action[4]:.1f}, {action[5]:.1f}]")
                logger.info(f"Sim Data: {sim_data}")
                
            if sim.time % config['timing']['visualization_frequency'] < sim.dt:
                Visualizer.plot_results(sim)
                plt.pause(0.001)
        
        logger.info("Simulation complete")
        
        # Save final figure
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
