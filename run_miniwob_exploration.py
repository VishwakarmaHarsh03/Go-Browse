#!/usr/bin/env python3
"""
MiniWob++ Exploration Runner

This script provides a convenient way to run MiniWob++ exploration
with different configurations and agents.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf as oc

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from webexp.explore.algorithms.miniwob_explore import MiniWobExplorer, MiniWobExploreConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('miniwob_exploration.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for MiniWob++ exploration."""
    # Check if required packages are installed
    try:
        import miniwob
        import gymnasium as gym
        logger.info("MiniWob++ environment is available")
    except ImportError as e:
        logger.error(f"MiniWob++ not installed: {e}")
        logger.error("Please install with: pip install miniwob")
        sys.exit(1)
    
    # Check Chrome setup
    try:
        from webexp.benchmark.run_miniwob import setup_chrome_environment
        setup_chrome_environment()
        logger.info("Chrome environment is properly set up")
    except Exception as e:
        logger.warning(f"Chrome setup issue: {e}")
        logger.warning("You may need to run the Chrome setup script")

def validate_config(config: MiniWobExploreConfig):
    """Validate the exploration configuration."""
    if not config.env_names:
        raise ValueError("No environments specified in config")
    
    if config.episodes_per_env <= 0:
        raise ValueError("episodes_per_env must be positive")
    
    if not config.exp_dir:
        raise ValueError("exp_dir must be specified")
    
    # Validate agent configurations
    if not config.explorer_agent.agent_factory_args:
        raise ValueError("Explorer agent configuration is missing")
    
    logger.info(f"Configuration validated: {len(config.env_names)} environments, "
               f"{config.episodes_per_env} episodes each")

def run_exploration(config_path: str, overrides: dict = None):
    """
    Run MiniWob++ exploration with the given configuration.
    
    Args:
        config_path (str): Path to the configuration file.
        overrides (dict): Configuration overrides.
    """
    logger.info(f"Loading configuration from: {config_path}")
    
    # Load configuration
    config_dict = oc.load(config_path)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            oc.set_struct(config_dict, False)  # Allow new keys
            oc.update(config_dict, key, value)
            oc.set_struct(config_dict, True)   # Re-enable struct mode
    
    # Create structured config
    config = oc.structured(MiniWobExploreConfig, config_dict)
    
    # Validate configuration
    validate_config(config)
    
    # Create and run explorer
    logger.info("Starting MiniWob++ exploration...")
    explorer = MiniWobExplorer(config)
    explorer.explore()
    
    logger.info("Exploration completed successfully!")

def list_available_environments():
    """List all available MiniWob++ environments."""
    try:
        import miniwob
        import gymnasium as gym
        
        # Get all MiniWob++ environments
        all_envs = gym.envs.registry.all()
        miniwob_envs = [env.id for env in all_envs if env.id.startswith('miniwob/')]
        
        # Clean up environment names (remove miniwob/ prefix and -v1 suffix)
        clean_names = []
        for env_id in miniwob_envs:
            clean_name = env_id.replace('miniwob/', '').replace('-v1', '')
            clean_names.append(clean_name)
        
        clean_names.sort()
        
        print("Available MiniWob++ environments:")
        print("=" * 40)
        for i, env_name in enumerate(clean_names, 1):
            print(f"{i:2d}. {env_name}")
        
        print(f"\nTotal: {len(clean_names)} environments")
        
    except ImportError:
        print("MiniWob++ not installed. Please install with: pip install miniwob")

def create_custom_config(output_path: str, env_names: list, episodes_per_env: int, 
                        agent_type: str = "azure"):
    """
    Create a custom exploration configuration.
    
    Args:
        output_path (str): Path to save the configuration.
        env_names (list): List of environment names.
        episodes_per_env (int): Number of episodes per environment.
        agent_type (str): Type of agent to use ("azure" or "bedrock").
    """
    if agent_type == "azure":
        agent_config = {
            "agent_name": "AzureGPTMiniWobAgent",
            "model_name": "gpt-4o",
            "temperature": 0.1,
            "char_limit": 40000,
            "demo_mode": "off"
        }
    elif agent_type == "bedrock":
        agent_config = {
            "agent_name": "BedrockClaudeMiniWobAgent",
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "temperature": 0.1,
            "max_tokens": 10000,
            "char_limit": 40000,
            "demo_mode": "off"
        }
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    config = {
        "env_names": env_names,
        "episodes_per_env": episodes_per_env,
        "explorer_agent": {
            "agent_factory_args": agent_config,
            "max_steps": 20,
            "retries": 3
        },
        "evaluator_agent": None,
        "exp_dir": f"./exploration_results/custom_{agent_type}",
        "headless": True,
        "slow_mo": 0,
        "viewport_size": {"width": 1280, "height": 720},
        "save_screenshots": True,
        "save_traces": True
    }
    
    # Save configuration
    oc.save(oc.create(config), output_path)
    logger.info(f"Custom configuration saved to: {output_path}")

def main():
    """Main function for the exploration runner."""
    parser = argparse.ArgumentParser(description="Run MiniWob++ exploration")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run exploration command
    run_parser = subparsers.add_parser("run", help="Run exploration")
    run_parser.add_argument("--config", type=str, required=True,
                           help="Path to exploration configuration file")
    run_parser.add_argument("--exp_dir", type=str, default=None,
                           help="Override experiment directory")
    run_parser.add_argument("--episodes_per_env", type=int, default=None,
                           help="Override episodes per environment")
    run_parser.add_argument("--env_names", type=str, nargs="+", default=None,
                           help="Override environment names")
    
    # List environments command
    list_parser = subparsers.add_parser("list", help="List available environments")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create custom configuration")
    config_parser.add_argument("--output", type=str, required=True,
                              help="Output path for the configuration file")
    config_parser.add_argument("--env_names", type=str, nargs="+", required=True,
                              help="List of environment names")
    config_parser.add_argument("--episodes_per_env", type=int, default=10,
                              help="Number of episodes per environment")
    config_parser.add_argument("--agent_type", type=str, choices=["azure", "bedrock"],
                              default="azure", help="Type of agent to use")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up environment")
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Prepare overrides
        overrides = {}
        if args.exp_dir:
            overrides["exp_dir"] = args.exp_dir
        if args.episodes_per_env:
            overrides["episodes_per_env"] = args.episodes_per_env
        if args.env_names:
            overrides["env_names"] = args.env_names
        
        # Set up environment
        setup_environment()
        
        # Run exploration
        run_exploration(args.config, overrides)
        
    elif args.command == "list":
        list_available_environments()
        
    elif args.command == "create-config":
        create_custom_config(
            output_path=args.output,
            env_names=args.env_names,
            episodes_per_env=args.episodes_per_env,
            agent_type=args.agent_type
        )
        
    elif args.command == "setup":
        setup_environment()
        print("Environment setup completed!")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()