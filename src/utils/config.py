import yaml
from pathlib import Path
from typing import Any

def Load_Config(config_path: str | Path, param: str = None) -> Any:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file (str or Path)
        param: Optional specific parameter to retrieve (using dot notation for nested keys)
        
    Returns:
        dict: Full configuration dictionary if param is None
        Any: Value of the specific parameter if param is provided
        
    Examples:
        config = load_config("config.yaml")  # get full config
        dt = load_config("config.yaml", "simulation.dt")  # get nested parameter
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if param is None:
        return config
        
    # Handle nested parameters using dot notation
    keys = param.split('.')
    value = config
    for key in keys:
        try:
            value = value[key]
        except (KeyError, TypeError):
            raise KeyError(f"Parameter '{param}' not found in config")
            
    return value