"""
MiniWob++ Supervised Fine-Tuning (SFT) Policy Training

This module provides training capabilities for MiniWob++ agents using
collected exploration trajectories.
"""

import os
import json
import torch
from datasets import Dataset, load_dataset
from datetime import timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.import_utils import is_torch_bf16_gpu_available
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import argparse
from omegaconf import OmegaConf as oc

logger = logging.getLogger(__name__)

class MiniWobSFTTrainer:
    """
    Supervised Fine-Tuning trainer for MiniWob++ agents.
    
    This trainer processes exploration trajectories and trains language models
    to perform MiniWob++ tasks through behavioral cloning.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "./outputs_miniwob_sft",
        max_seq_length: int = 8192,
        wandb_project: str = "Go-Browse-MiniWob",
        wandb_api_key: Optional[str] = None
    ):
        """
        Initialize the MiniWob++ SFT trainer.
        
        Args:
            model_id (str): Hugging Face model identifier.
            output_dir (str): Directory to save trained models.
            max_seq_length (int): Maximum sequence length for training.
            wandb_project (str): Weights & Biases project name.
            wandb_api_key (Optional[str]): Weights & Biases API key.
        """
        self.model_id = model_id
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.wandb_project = wandb_project
        
        # Set up Weights & Biases
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_PROJECT"] = wandb_project
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"MiniWob++ SFT Trainer initialized")
        logger.info(f"Model: {model_id}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Max sequence length: {max_seq_length}")
    
    def prepare_dataset_from_exploration(
        self,
        exploration_dir: str,
        output_path: str,
        include_negative: bool = False,
        min_success_rate: float = 0.0
    ) -> str:
        """
        Prepare training dataset from exploration results.
        
        Args:
            exploration_dir (str): Directory containing exploration results.
            output_path (str): Path to save the prepared dataset.
            include_negative (bool): Whether to include failed trajectories.
            min_success_rate (float): Minimum success rate for including an environment.
            
        Returns:
            str: Path to the prepared dataset file.
        """
        logger.info(f"Preparing dataset from exploration directory: {exploration_dir}")
        
        training_examples = []
        env_stats = {}
        
        # Process each environment directory
        for env_dir in os.listdir(exploration_dir):
            env_path = os.path.join(exploration_dir, env_dir)
            if not os.path.isdir(env_path):
                continue
                
            # Skip non-environment directories
            if env_dir in ["exploration_summary.json", "explore_config.yaml"]:
                continue
            
            logger.info(f"Processing environment: {env_dir}")
            
            # Load task info
            task_info_path = os.path.join(env_path, "task_info.json")
            if os.path.exists(task_info_path):
                with open(task_info_path, "r") as f:
                    task_info = json.load(f)
                env_name = task_info.get("env_name", env_dir)
            else:
                env_name = env_dir
            
            # Count trajectories and calculate success rate
            positive_dir = os.path.join(env_path, "positive_trajs")
            negative_dir = os.path.join(env_path, "negative_trajs")
            
            positive_count = len(os.listdir(positive_dir)) if os.path.exists(positive_dir) else 0
            negative_count = len(os.listdir(negative_dir)) if os.path.exists(negative_dir) else 0
            total_count = positive_count + negative_count
            
            if total_count == 0:
                logger.warning(f"No trajectories found for {env_name}")
                continue
                
            success_rate = positive_count / total_count
            env_stats[env_name] = {
                "positive": positive_count,
                "negative": negative_count,
                "total": total_count,
                "success_rate": success_rate
            }
            
            # Skip environments with low success rate
            if success_rate < min_success_rate:
                logger.info(f"Skipping {env_name} (success rate: {success_rate:.2%} < {min_success_rate:.2%})")
                continue
            
            # Process positive trajectories
            if os.path.exists(positive_dir):
                for traj_dir in os.listdir(positive_dir):
                    traj_path = os.path.join(positive_dir, traj_dir)
                    if os.path.isdir(traj_path):
                        example = self._process_trajectory(env_name, traj_path, success=True)
                        if example:
                            training_examples.append(example)
            
            # Process negative trajectories if requested
            if include_negative and os.path.exists(negative_dir):
                for traj_dir in os.listdir(negative_dir):
                    traj_path = os.path.join(negative_dir, traj_dir)
                    if os.path.isdir(traj_path):
                        example = self._process_trajectory(env_name, traj_path, success=False)
                        if example:
                            training_examples.append(example)
        
        # Save dataset
        logger.info(f"Prepared {len(training_examples)} training examples from {len(env_stats)} environments")
        
        # Save as JSONL
        with open(output_path, "w") as f:
            for example in training_examples:
                f.write(json.dumps(example) + "\n")
        
        # Save statistics
        stats_path = output_path.replace(".jsonl", "_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "total_examples": len(training_examples),
                "environments": env_stats,
                "config": {
                    "include_negative": include_negative,
                    "min_success_rate": min_success_rate
                }
            }, f, indent=2)
        
        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"Statistics saved to: {stats_path}")
        
        return output_path
    
    def _process_trajectory(self, env_name: str, traj_path: str, success: bool) -> Optional[Dict]:
        """
        Process a single trajectory into a training example.
        
        Args:
            env_name (str): Name of the environment.
            traj_path (str): Path to the trajectory directory.
            success (bool): Whether the trajectory was successful.
            
        Returns:
            Optional[Dict]: Training example or None if processing failed.
        """
        try:
            # Load trajectory data
            traj_file = os.path.join(traj_path, "trajectory.json")
            if not os.path.exists(traj_file):
                return None
                
            with open(traj_file, "r") as f:
                traj_data = json.load(f)
            
            # Extract steps
            steps = traj_data.get("steps", [])
            if not steps:
                return None
            
            # Build conversation format
            conversation = self._build_conversation(env_name, steps, success)
            
            return {
                "env_name": env_name,
                "success": success,
                "num_steps": len(steps),
                "text": conversation,
                "trajectory_path": traj_path
            }
            
        except Exception as e:
            logger.warning(f"Failed to process trajectory {traj_path}: {e}")
            return None
    
    def _build_conversation(self, env_name: str, steps: List[Dict], success: bool) -> str:
        """
        Build a conversation format from trajectory steps.
        
        Args:
            env_name (str): Name of the environment.
            steps (List[Dict]): List of trajectory steps.
            success (bool): Whether the trajectory was successful.
            
        Returns:
            str: Formatted conversation string.
        """
        # System prompt
        system_prompt = f"""You are an AI assistant that can interact with web interfaces to complete tasks. You are currently working on the MiniWob++ task: {env_name}.

Your goal is to complete the task by taking appropriate actions based on the current state of the interface. You can:
- click(element_id): Click on an element
- type(text): Type text into the current input field
- key(key_name): Press a specific key
- scroll(direction): Scroll in a direction

Analyze the current state and take the most appropriate action to complete the task."""

        # Build conversation
        conversation = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        for i, step in enumerate(steps):
            # Add observation
            obs = step.get("observation", {})
            if isinstance(obs, dict):
                # Extract relevant observation information
                obs_text = self._format_observation(obs)
                conversation += f"<|im_start|>user\nCurrent state:\n{obs_text}<|im_end|>\n"
            
            # Add action
            action = step.get("action", "")
            if action:
                conversation += f"<|im_start|>assistant\n{action}<|im_end|>\n"
        
        return conversation
    
    def _format_observation(self, obs: Dict) -> str:
        """
        Format observation dictionary into readable text.
        
        Args:
            obs (Dict): Observation dictionary.
            
        Returns:
            str: Formatted observation text.
        """
        formatted_parts = []
        
        # Add DOM information if available
        if "dom_elements" in obs:
            formatted_parts.append("DOM Elements:")
            for element in obs["dom_elements"][:10]:  # Limit to first 10 elements
                formatted_parts.append(f"- {element}")
        
        # Add text content if available
        if "text" in obs:
            formatted_parts.append(f"Text content: {obs['text']}")
        
        # Add goal if available
        if "goal" in obs:
            formatted_parts.append(f"Goal: {obs['goal']}")
        
        # Add any other relevant fields
        for key, value in obs.items():
            if key not in ["dom_elements", "text", "goal", "screenshot"] and isinstance(value, str):
                formatted_parts.append(f"{key}: {value}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No observation data available"
    
    def train(
        self,
        dataset_path: str,
        num_train_epochs: int = 2,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 10
    ):
        """
        Train the model using the prepared dataset.
        
        Args:
            dataset_path (str): Path to the training dataset.
            num_train_epochs (int): Number of training epochs.
            per_device_train_batch_size (int): Batch size per device.
            gradient_accumulation_steps (int): Gradient accumulation steps.
            learning_rate (float): Learning rate.
            warmup_steps (int): Number of warmup steps.
            save_steps (int): Save model every N steps.
            logging_steps (int): Log every N steps.
        """
        logger.info("Starting MiniWob++ SFT training...")
        
        # Load dataset
        logger.info(f"Loading dataset from: {dataset_path}")
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        
        # Load tokenizer and model
        logger.info(f"Loading model and tokenizer: {self.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Training configuration
        training_config = SFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=not is_torch_bf16_gpu_available(),
            bf16=is_torch_bf16_gpu_available(),
            logging_steps=logging_steps,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=self.output_dir,
            save_strategy="steps",
            save_steps=save_steps,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            packing=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            save_total_limit=3,
            dataset_num_proc=4,
            report_to=["wandb"] if "WANDB_API_KEY" in os.environ else [],
            model_init_kwargs={
                "attn_implementation": "flash_attention_2",
                "trust_remote_code": True,
            }
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model_id,
            args=training_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed! Model saved to: {self.output_dir}")

def main():
    """Main function for running MiniWob++ SFT training from command line."""
    parser = argparse.ArgumentParser(description="Train MiniWob++ SFT policy")
    parser.add_argument("--exploration_dir", type=str, required=True,
                       help="Directory containing exploration results")
    parser.add_argument("--output_dir", type=str, default="./outputs_miniwob_sft",
                       help="Output directory for trained model")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Hugging Face model identifier")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Path to prepared dataset (if not provided, will be created)")
    parser.add_argument("--include_negative", action="store_true",
                       help="Include failed trajectories in training")
    parser.add_argument("--min_success_rate", type=float, default=0.0,
                       help="Minimum success rate for including an environment")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                       help="Weights & Biases API key")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MiniWobSFTTrainer(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        wandb_api_key=args.wandb_api_key
    )
    
    # Prepare dataset if not provided
    if args.dataset_path is None:
        dataset_path = os.path.join(args.output_dir, "training_dataset.jsonl")
        trainer.prepare_dataset_from_exploration(
            exploration_dir=args.exploration_dir,
            output_path=dataset_path,
            include_negative=args.include_negative,
            min_success_rate=args.min_success_rate
        )
    else:
        dataset_path = args.dataset_path
    
    # Train model
    trainer.train(
        dataset_path=dataset_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()