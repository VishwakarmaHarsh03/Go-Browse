"""
MiniWob++ Task Management

This module provides task management specifically for MiniWob++ environments.
It handles task creation, trajectory storage, and data organization for training.
"""

from ..core.trajectory import Trajectory
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

@dataclass
class MiniWobTask:
    """
    Represents a MiniWob++ task with collected trajectories.
    
    Unlike WebArena tasks which are goal-based, MiniWob++ tasks are environment-based
    where each environment represents a specific UI interaction task.
    """
    env_name: str
    exp_dir: str
    config: Any
    positive_trajs: List[Trajectory] = None
    negative_trajs: List[Trajectory] = None
    misc: Dict = None
    
    def __post_init__(self):
        if self.positive_trajs is None:
            self.positive_trajs = []
        if self.negative_trajs is None:
            self.negative_trajs = []
        if self.misc is None:
            self.misc = {}
            
        # Create directory structure
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
            os.makedirs(os.path.join(self.exp_dir, "positive_trajs"))
            os.makedirs(os.path.join(self.exp_dir, "negative_trajs"))
            os.makedirs(os.path.join(self.exp_dir, "episodes"))
            
            # Save task information
            task_info = {
                "env_name": self.env_name,
                "created_at": time.time(),
                "config": {
                    "max_steps": getattr(self.config.explorer_agent, 'max_steps', None),
                    "headless": getattr(self.config, 'headless', True),
                    "viewport_size": getattr(self.config, 'viewport_size', None)
                },
                "misc": self.misc,
            }
            with open(os.path.join(self.exp_dir, "task_info.json"), "w") as f:
                json.dump(task_info, f, indent=4)

    def is_feasible(self) -> bool:
        """Check if the task has any successful trajectories."""
        return len(self.positive_trajs) > 0
    
    def get_success_rate(self) -> float:
        """Calculate the success rate for this task."""
        total = len(self.positive_trajs) + len(self.negative_trajs)
        if total == 0:
            return 0.0
        return len(self.positive_trajs) / total
    
    def add_trajectory(self, traj: Trajectory, subdirectory: Optional[str] = None):
        """
        Add a trajectory to the task.
        
        Args:
            traj (Trajectory): The trajectory to add.
            subdirectory (Optional[str]): Optional subdirectory for organization.
        """
        exp_dir = self.exp_dir
        
        if subdirectory is not None:
            exp_dir = os.path.join(exp_dir, subdirectory)
            os.makedirs(os.path.join(exp_dir, "positive_trajs"), exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "negative_trajs"), exist_ok=True)
        
        if traj.success:
            traj_save_dir = os.path.join(exp_dir, "positive_trajs", f"{len(self.positive_trajs)}")
            self.positive_trajs.append(traj)
            logger.info(f"Saving positive trajectory for {self.env_name} to {traj_save_dir}")
            os.makedirs(traj_save_dir, exist_ok=True)
            traj.save(traj_save_dir)
        else:
            traj_save_dir = os.path.join(exp_dir, "negative_trajs", f"{len(self.negative_trajs)}")
            self.negative_trajs.append(traj)
            logger.info(f"Saving negative trajectory for {self.env_name} to {traj_save_dir}")
            os.makedirs(traj_save_dir, exist_ok=True)
            traj.save(traj_save_dir)
    
    def get_trajectory_stats(self) -> Dict[str, Any]:
        """Get statistics about the collected trajectories."""
        stats = {
            "env_name": self.env_name,
            "total_trajectories": len(self.positive_trajs) + len(self.negative_trajs),
            "successful_trajectories": len(self.positive_trajs),
            "failed_trajectories": len(self.negative_trajs),
            "success_rate": self.get_success_rate(),
        }
        
        if self.positive_trajs:
            positive_steps = [len(traj.steps) for traj in self.positive_trajs]
            stats["positive_trajectory_stats"] = {
                "avg_steps": sum(positive_steps) / len(positive_steps),
                "min_steps": min(positive_steps),
                "max_steps": max(positive_steps),
            }
        
        if self.negative_trajs:
            negative_steps = [len(traj.steps) for traj in self.negative_trajs]
            stats["negative_trajectory_stats"] = {
                "avg_steps": sum(negative_steps) / len(negative_steps),
                "min_steps": min(negative_steps),
                "max_steps": max(negative_steps),
            }
        
        return stats
    
    def export_for_training(self, output_dir: str, format: str = "jsonl"):
        """
        Export trajectories in a format suitable for training.
        
        Args:
            output_dir (str): Directory to save the training data.
            format (str): Export format ('jsonl', 'json', or 'hf_dataset').
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if format == "jsonl":
            self._export_jsonl(output_dir)
        elif format == "json":
            self._export_json(output_dir)
        elif format == "hf_dataset":
            self._export_hf_dataset(output_dir)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_jsonl(self, output_dir: str):
        """Export trajectories as JSONL files."""
        # Export positive trajectories
        positive_file = os.path.join(output_dir, f"{self.env_name}_positive.jsonl")
        with open(positive_file, "w") as f:
            for i, traj in enumerate(self.positive_trajs):
                data = {
                    "env_name": self.env_name,
                    "trajectory_id": i,
                    "success": True,
                    "steps": len(traj.steps),
                    "final_reward": traj.final_reward,
                    "trajectory": traj.to_dict()
                }
                f.write(json.dumps(data) + "\n")
        
        # Export negative trajectories
        negative_file = os.path.join(output_dir, f"{self.env_name}_negative.jsonl")
        with open(negative_file, "w") as f:
            for i, traj in enumerate(self.negative_trajs):
                data = {
                    "env_name": self.env_name,
                    "trajectory_id": i,
                    "success": False,
                    "steps": len(traj.steps),
                    "final_reward": traj.final_reward,
                    "trajectory": traj.to_dict()
                }
                f.write(json.dumps(data) + "\n")
        
        logger.info(f"Exported {len(self.positive_trajs)} positive and {len(self.negative_trajs)} "
                   f"negative trajectories for {self.env_name} to {output_dir}")
    
    def _export_json(self, output_dir: str):
        """Export trajectories as JSON files."""
        data = {
            "env_name": self.env_name,
            "stats": self.get_trajectory_stats(),
            "positive_trajectories": [traj.to_dict() for traj in self.positive_trajs],
            "negative_trajectories": [traj.to_dict() for traj in self.negative_trajs]
        }
        
        output_file = os.path.join(output_dir, f"{self.env_name}_trajectories.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported trajectories for {self.env_name} to {output_file}")
    
    def _export_hf_dataset(self, output_dir: str):
        """Export trajectories as Hugging Face dataset format."""
        try:
            from datasets import Dataset
            
            # Prepare data for HF dataset
            data = []
            
            # Add positive trajectories
            for i, traj in enumerate(self.positive_trajs):
                data.append({
                    "env_name": self.env_name,
                    "trajectory_id": f"positive_{i}",
                    "success": True,
                    "steps": len(traj.steps),
                    "final_reward": traj.final_reward,
                    "trajectory": json.dumps(traj.to_dict())
                })
            
            # Add negative trajectories
            for i, traj in enumerate(self.negative_trajs):
                data.append({
                    "env_name": self.env_name,
                    "trajectory_id": f"negative_{i}",
                    "success": False,
                    "steps": len(traj.steps),
                    "final_reward": traj.final_reward,
                    "trajectory": json.dumps(traj.to_dict())
                })
            
            # Create dataset
            dataset = Dataset.from_list(data)
            
            # Save dataset
            dataset_path = os.path.join(output_dir, f"{self.env_name}_dataset")
            dataset.save_to_disk(dataset_path)
            
            logger.info(f"Exported HF dataset for {self.env_name} to {dataset_path}")
            
        except ImportError:
            logger.error("Hugging Face datasets library not available. Please install with: pip install datasets")
            raise
    
    @classmethod
    def load_from_directory(cls, exp_dir: str, config: Any = None):
        """
        Load a MiniWobTask from a saved directory.
        
        Args:
            exp_dir (str): Directory containing the saved task.
            config (Any): Configuration object (optional).
            
        Returns:
            MiniWobTask: Loaded task object.
        """
        # Load task info
        task_info_path = os.path.join(exp_dir, "task_info.json")
        if os.path.exists(task_info_path):
            with open(task_info_path, "r") as f:
                task_info = json.load(f)
            env_name = task_info["env_name"]
            misc = task_info.get("misc", {})
        else:
            # Fallback: extract env_name from directory name
            env_name = os.path.basename(exp_dir)
            misc = {}
        
        # Create task
        task = cls(
            env_name=env_name,
            exp_dir=exp_dir,
            config=config,
            misc=misc
        )
        
        # Load trajectories
        positive_dir = os.path.join(exp_dir, "positive_trajs")
        negative_dir = os.path.join(exp_dir, "negative_trajs")
        
        if os.path.exists(positive_dir):
            for traj_dir in sorted(os.listdir(positive_dir)):
                traj_path = os.path.join(positive_dir, traj_dir)
                if os.path.isdir(traj_path):
                    try:
                        traj = Trajectory.load(traj_path)
                        task.positive_trajs.append(traj)
                    except Exception as e:
                        logger.warning(f"Failed to load positive trajectory from {traj_path}: {e}")
        
        if os.path.exists(negative_dir):
            for traj_dir in sorted(os.listdir(negative_dir)):
                traj_path = os.path.join(negative_dir, traj_dir)
                if os.path.isdir(traj_path):
                    try:
                        traj = Trajectory.load(traj_path)
                        task.negative_trajs.append(traj)
                    except Exception as e:
                        logger.warning(f"Failed to load negative trajectory from {traj_path}: {e}")
        
        logger.info(f"Loaded MiniWobTask for {env_name}: "
                   f"{len(task.positive_trajs)} positive, {len(task.negative_trajs)} negative trajectories")
        
        return task