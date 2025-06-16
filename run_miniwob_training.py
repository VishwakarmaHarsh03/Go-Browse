#!/usr/bin/env python3
"""
MiniWob++ Training Runner

This script provides a convenient way to train models on collected
MiniWob++ exploration data.
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

from webexp.train.miniwob_sft_policy import MiniWobSFTTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('miniwob_training.log')
    ]
)

logger = logging.getLogger(__name__)

def check_training_requirements():
    """Check if all required packages for training are installed."""
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {name} is available")
        except ImportError:
            missing_packages.append((package, name))
            logger.error(f"✗ {name} is not installed")
    
    if missing_packages:
        logger.error("Missing required packages for training:")
        for package, name in missing_packages:
            logger.error(f"  - {name}: pip install {package}")
        return False
    
    return True

def validate_exploration_data(exploration_dir: str):
    """Validate that exploration data is available and properly formatted."""
    if not os.path.exists(exploration_dir):
        raise ValueError(f"Exploration directory does not exist: {exploration_dir}")
    
    # Check for exploration summary
    summary_path = os.path.join(exploration_dir, "exploration_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        total_episodes = summary.get("total_episodes", 0)
        success_rate = summary.get("overall_success_rate", 0)
        
        logger.info(f"Found exploration data: {total_episodes} episodes, "
                   f"{success_rate:.2%} success rate")
        
        if total_episodes == 0:
            raise ValueError("No episodes found in exploration data")
        
        if success_rate == 0:
            logger.warning("No successful episodes found - training may not be effective")
    
    # Check for environment directories
    env_dirs = [d for d in os.listdir(exploration_dir) 
                if os.path.isdir(os.path.join(exploration_dir, d)) 
                and d not in ["exploration_summary.json", "explore_config.yaml"]]
    
    if not env_dirs:
        raise ValueError("No environment directories found in exploration data")
    
    logger.info(f"Found {len(env_dirs)} environment directories")
    
    # Check for trajectories in each environment
    total_positive = 0
    total_negative = 0
    
    for env_dir in env_dirs:
        env_path = os.path.join(exploration_dir, env_dir)
        positive_dir = os.path.join(env_path, "positive_trajs")
        negative_dir = os.path.join(env_path, "negative_trajs")
        
        positive_count = len(os.listdir(positive_dir)) if os.path.exists(positive_dir) else 0
        negative_count = len(os.listdir(negative_dir)) if os.path.exists(negative_dir) else 0
        
        total_positive += positive_count
        total_negative += negative_count
        
        logger.info(f"  {env_dir}: {positive_count} positive, {negative_count} negative")
    
    logger.info(f"Total trajectories: {total_positive} positive, {total_negative} negative")
    
    if total_positive == 0:
        raise ValueError("No positive trajectories found - cannot train without successful examples")

def run_training(config_path: str, overrides: dict = None):
    """
    Run MiniWob++ training with the given configuration.
    
    Args:
        config_path (str): Path to the training configuration file.
        overrides (dict): Configuration overrides.
    """
    logger.info(f"Loading training configuration from: {config_path}")
    
    # Load configuration
    config = oc.load(config_path)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            oc.set_struct(config, False)  # Allow new keys
            oc.update(config, key, value)
            oc.set_struct(config, True)   # Re-enable struct mode
    
    # Validate exploration data
    validate_exploration_data(config.exploration_dir)
    
    # Create trainer
    trainer = MiniWobSFTTrainer(
        model_id=config.model_id,
        output_dir=config.output_dir,
        max_seq_length=config.max_seq_length,
        wandb_project=config.wandb.project,
        wandb_api_key=os.environ.get("WANDB_API_KEY")
    )
    
    # Prepare dataset
    dataset_path = os.path.join(config.output_dir, "training_dataset.jsonl")
    logger.info("Preparing training dataset...")
    
    trainer.prepare_dataset_from_exploration(
        exploration_dir=config.exploration_dir,
        output_path=dataset_path,
        include_negative=config.include_negative,
        min_success_rate=config.min_success_rate
    )
    
    # Train model
    logger.info("Starting model training...")
    trainer.train(
        dataset_path=dataset_path,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        save_steps=config.logging.save_steps,
        logging_steps=config.logging.logging_steps
    )
    
    logger.info("Training completed successfully!")

def analyze_exploration_data(exploration_dir: str):
    """Analyze exploration data and provide insights for training."""
    logger.info(f"Analyzing exploration data in: {exploration_dir}")
    
    # Load exploration summary if available
    summary_path = os.path.join(exploration_dir, "exploration_summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        print("\n" + "="*50)
        print("EXPLORATION DATA ANALYSIS")
        print("="*50)
        
        print(f"Total Episodes: {summary.get('total_episodes', 0)}")
        print(f"Successful Episodes: {summary.get('total_successful_episodes', 0)}")
        print(f"Failed Episodes: {summary.get('total_failed_episodes', 0)}")
        print(f"Overall Success Rate: {summary.get('overall_success_rate', 0):.2%}")
        
        print("\nPer-Environment Breakdown:")
        print("-" * 30)
        
        environments = summary.get("environments", {})
        sorted_envs = sorted(environments.items(), 
                           key=lambda x: x[1]["success_rate"], reverse=True)
        
        for env_name, stats in sorted_envs:
            success_rate = stats["success_rate"]
            total = stats["total_episodes"]
            successful = stats["successful_episodes"]
            
            status = "✓" if success_rate > 0.5 else "⚠" if success_rate > 0.1 else "✗"
            print(f"{status} {env_name:20s} {successful:2d}/{total:2d} ({success_rate:5.1%})")
        
        # Recommendations
        print("\nTraining Recommendations:")
        print("-" * 25)
        
        high_success_envs = [env for env, stats in environments.items() 
                           if stats["success_rate"] > 0.5]
        medium_success_envs = [env for env, stats in environments.items() 
                             if 0.1 < stats["success_rate"] <= 0.5]
        low_success_envs = [env for env, stats in environments.items() 
                          if stats["success_rate"] <= 0.1]
        
        if high_success_envs:
            print(f"• {len(high_success_envs)} environments with >50% success rate - excellent for training")
        if medium_success_envs:
            print(f"• {len(medium_success_envs)} environments with 10-50% success rate - good for training")
        if low_success_envs:
            print(f"• {len(low_success_envs)} environments with <10% success rate - consider excluding")
        
        recommended_min_success = 0.1 if len(high_success_envs) + len(medium_success_envs) > 5 else 0.0
        print(f"• Recommended min_success_rate: {recommended_min_success}")
        
        total_positive = sum(stats["successful_episodes"] for stats in environments.values())
        if total_positive < 50:
            print("• Warning: Low number of successful episodes - consider collecting more data")
        elif total_positive < 200:
            print("• Moderate amount of training data - should be sufficient for initial training")
        else:
            print("• Good amount of training data available")
    
    else:
        print("No exploration summary found. Running basic analysis...")
        validate_exploration_data(exploration_dir)

def main():
    """Main function for the training runner."""
    parser = argparse.ArgumentParser(description="Run MiniWob++ training")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", type=str, required=True,
                             help="Path to training configuration file")
    train_parser.add_argument("--exploration_dir", type=str, default=None,
                             help="Override exploration directory")
    train_parser.add_argument("--output_dir", type=str, default=None,
                             help="Override output directory")
    train_parser.add_argument("--model_id", type=str, default=None,
                             help="Override model ID")
    train_parser.add_argument("--epochs", type=int, default=None,
                             help="Override number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=None,
                             help="Override batch size")
    train_parser.add_argument("--learning_rate", type=float, default=None,
                             help="Override learning rate")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze exploration data")
    analyze_parser.add_argument("exploration_dir", type=str,
                               help="Directory containing exploration results")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check training requirements")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # Check requirements
        if not check_training_requirements():
            sys.exit(1)
        
        # Prepare overrides
        overrides = {}
        if args.exploration_dir:
            overrides["exploration_dir"] = args.exploration_dir
        if args.output_dir:
            overrides["output_dir"] = args.output_dir
        if args.model_id:
            overrides["model_id"] = args.model_id
        if args.epochs:
            overrides["training.num_train_epochs"] = args.epochs
        if args.batch_size:
            overrides["training.per_device_train_batch_size"] = args.batch_size
        if args.learning_rate:
            overrides["training.learning_rate"] = args.learning_rate
        
        # Run training
        run_training(args.config, overrides)
        
    elif args.command == "analyze":
        analyze_exploration_data(args.exploration_dir)
        
    elif args.command == "check":
        if check_training_requirements():
            print("✓ All training requirements are satisfied!")
        else:
            print("✗ Some training requirements are missing")
            sys.exit(1)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()