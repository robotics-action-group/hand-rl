from yaml import safe_load
import os
# import sys


def load_yaml_into_obj(yaml_file_path, obj):
    """
    Load a YAML configuration file.
    
    Args:
        yaml_file_path (str): Path to the YAML file.
        obj (object): Object to update with the YAML content.        
    Returns:
        object: Updated object.
    """
    if not os.path.exists(yaml_file_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_file_path}")
    
    with open(yaml_file_path, 'r') as file:
        config = safe_load(file)
    
    obj = update_obj_from_dict(obj, config)
    
    return obj

def update_obj_from_dict(obj, config):
    """
    Update an object with values from a dictionary.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recursively update the object
            nested_obj = getattr(obj, key, None)
            if nested_obj is not None:
                update_obj_from_dict(nested_obj, value)
        elif hasattr(obj, key):
            setattr(obj, key, value)
    return obj