import argparse
import os
import sys
import pathlib
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

dreamer = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer)
sys.path.append(str(pathlib.Path(__file__).parent))
import tools

def load_config(config_path=None, config_list=None):
    """
    Load configuration using Hydra.
    
    Args:
        config_path: Path to config file (for backward compatibility, can be None to use default)
        config_list: List of additional config groups to load (for backward compatibility)
    
    Returns:
        argparse.Namespace-like object with config values as attributes
    """
    # Get the configs directory path
    configs_dir = pathlib.Path(__file__).parent.parent / "configs"
    
    # Initialize Hydra with the configs directory
    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        # Compose the config
        if config_list:
            # If config_list is provided, compose with additional configs
            cfg = compose(config_name="config", overrides=config_list)
        else:
            cfg = compose(config_name="config")
    
    # Convert OmegaConf to a flat dictionary, handling nested structures
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Convert to regular dict and flatten nested structures
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_config = flatten_dict(config_dict)
    
    # Create argparse.Namespace for backward compatibility
    parser = argparse.ArgumentParser()
    for key, value in sorted(flat_config.items(), key=lambda x: x[0]):
        # Convert hyphens to underscores for Python attribute access
        python_key = key.replace('-', '_')
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{python_key}", type=arg_type, default=arg_type(value))
    
    args = parser.parse_args([])
    
    # Also add original hyphenated keys for backward compatibility
    for key, value in flat_config.items():
        python_key = key.replace('-', '_')
        if not hasattr(args, python_key):
            setattr(args, python_key, value)
    
    return args

if __name__ == "__main__":
    config = load_config()
    print(config.dt)
