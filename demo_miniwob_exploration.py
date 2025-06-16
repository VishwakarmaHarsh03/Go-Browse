#!/usr/bin/env python3
"""
Demo script for MiniWob++ exploration functionality.

This script demonstrates how to run a simple exploration session
on a few MiniWob++ environments.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from webexp.explore.algorithms.miniwob_explore import MiniWobExplorer, MiniWobExploreConfig, MiniWobExploreAgentConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_demo_config():
    """Create a simple demo configuration."""
    
    # Agent configuration for Azure GPT
    agent_config = MiniWobExploreAgentConfig(
        agent_factory_args={
            "agent_name": "AzureGPTMiniWobAgent",
            "model_name": "gpt-4o",
            "temperature": 0.1,
            "char_limit": 40000,
            "demo_mode": "off"
        },
        max_steps=15,
        retries=2
    )
    
    # Exploration configuration
    config = MiniWobExploreConfig(
        env_names=[
            "click-test",
            "click-button", 
            "text-input"
        ],
        episodes_per_env=3,  # Small number for demo
        explorer_agent=agent_config,
        evaluator_agent=None,  # No evaluator for demo
        exp_dir="./demo_exploration_results",
        headless=True,
        slow_mo=0,
        viewport_size={"width": 1280, "height": 720},
        save_screenshots=True,
        save_traces=True
    )
    
    return config

def run_demo():
    """Run the exploration demo."""
    logger.info("Starting MiniWob++ exploration demo...")
    
    # Check if required packages are available
    try:
        import miniwob
        import gymnasium as gym
        logger.info("✓ MiniWob++ is available")
    except ImportError as e:
        logger.error(f"✗ MiniWob++ not available: {e}")
        logger.error("Please install with: pip install miniwob")
        return False
    
    # Check Chrome setup
    try:
        from webexp.benchmark.run_miniwob import setup_chrome_environment
        setup_chrome_environment()
        logger.info("✓ Chrome environment is set up")
    except Exception as e:
        logger.warning(f"Chrome setup issue: {e}")
        logger.warning("You may need to run: python setup_chrome.py")
    
    # Create configuration
    config = create_demo_config()
    logger.info(f"Demo configuration created:")
    logger.info(f"  - Environments: {config.env_names}")
    logger.info(f"  - Episodes per env: {config.episodes_per_env}")
    logger.info(f"  - Output directory: {config.exp_dir}")
    
    # Create and run explorer
    try:
        explorer = MiniWobExplorer(config)
        explorer.explore()
        
        logger.info("Demo completed successfully!")
        logger.info(f"Results saved to: {config.exp_dir}")
        
        # Show summary
        show_demo_results(config.exp_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def show_demo_results(exp_dir: str):
    """Show a summary of the demo results."""
    import json
    
    summary_path = os.path.join(exp_dir, "exploration_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        print("\n" + "="*40)
        print("DEMO RESULTS SUMMARY")
        print("="*40)
        
        print(f"Total Episodes: {summary.get('total_episodes', 0)}")
        print(f"Successful Episodes: {summary.get('total_successful_episodes', 0)}")
        print(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.1%}")
        
        print("\nPer-Environment Results:")
        environments = summary.get("environments", {})
        for env_name, stats in environments.items():
            success_rate = stats["success_rate"]
            total = stats["total_episodes"]
            successful = stats["successful_episodes"]
            
            status = "✓" if success_rate > 0 else "✗"
            print(f"  {status} {env_name}: {successful}/{total} ({success_rate:.1%})")
        
        print(f"\nDetailed results available in: {exp_dir}")
        
        # Show next steps
        print("\nNext Steps:")
        print("1. Analyze results: python run_miniwob_training.py analyze " + exp_dir)
        print("2. Run more exploration: python run_miniwob_exploration.py run --config configs/miniwob_explore_config.yaml")
        print("3. Train a model: python run_miniwob_training.py train --config configs/miniwob_train_config.yaml")
    
    else:
        print(f"No summary found in {exp_dir}")

def main():
    """Main function."""
    print("MiniWob++ Exploration Demo")
    print("=" * 30)
    print()
    print("This demo will:")
    print("1. Check that MiniWob++ and Chrome are set up")
    print("2. Run 3 episodes each on 3 simple environments")
    print("3. Show the results")
    print()
    
    # Ask for confirmation
    try:
        response = input("Continue with demo? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Demo cancelled.")
            return
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return
    
    # Run demo
    success = run_demo()
    
    if success:
        print("\n✓ Demo completed successfully!")
        print("Check the results in ./demo_exploration_results/")
    else:
        print("\n✗ Demo failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()