import os
import yaml
# Function to load YAML config file
def load_config(config_path='config/config.yaml'):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model_config():
    with open('config/model_config.yaml', 'r') as f:
        return yaml.safe_load(f)