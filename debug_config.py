#!/usr/bin/env python3
"""
Debug script to test configuration loading and overrides.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading with overrides."""
    
    # Load base config
    config_path = "configs/miniwob_explore_config.yaml"
    logger.info(f"Loading config from: {config_path}")
    
    try:
        config_dict = OmegaConf.load(config_path)
        logger.info(f"Base config loaded successfully")
        logger.info(f"Base env_names: {config_dict.get('env_names', 'NOT_FOUND')}")
        
        # Test overrides
        overrides = {"env_names": ["click-test"]}
        logger.info(f"Applying overrides: {overrides}")
        
        for key, value in overrides.items():
            OmegaConf.set_struct(config_dict, False)  # Allow new keys
            OmegaConf.update(config_dict, key, value)
            OmegaConf.set_struct(config_dict, True)   # Re-enable struct mode
        
        logger.info(f"After overrides env_names: {config_dict.get('env_names', 'NOT_FOUND')}")
        logger.info(f"Config dict keys: {list(config_dict.keys())}")
        
        # Test validation
        env_names = config_dict.get('env_names', [])
        logger.info(f"Validation - env_names: {env_names}, type: {type(env_names)}")
        
        if not env_names:
            logger.error("env_names is empty!")
        else:
            logger.info("env_names validation passed!")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_loading()