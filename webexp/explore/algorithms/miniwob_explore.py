"""
MiniWob++ Exploration Algorithm

This module provides exploration capabilities specifically designed for MiniWob++ environments.
Unlike WebArena's web browsing tasks, MiniWob++ consists of discrete UI interaction tasks.
"""

from ..core.agent import AgentWithExplorationCallbacks, ExplorerAgentWithExplorationCallbacks, wrap_agent_for_callback_protocol
from ..core.evaluator import Evaluator
from ..core.graph import Graph
from ..core.node import Node
from ..core.task import Task
from ..core.trace import Trace
from ..core.trajectory import Trajectory
from .miniwob_episode import run_miniwob_episode, MiniWobEpisodeResult
from .miniwob_task import MiniWobTask
from ...agents.base_agent import AgentFactory
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
from pathlib import Path
from typing import Sequence, List, Dict, Optional, Any
import argparse
import logging
import os
import random
import traceback
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class MiniWobExploreAgentConfig:
    """
    Configuration for MiniWob++ Explorer agents.

    Attributes:
        agent_factory_args (Dict): Arguments for the agent factory.
        max_steps (int): Maximum steps for the agent.
        retries (int): Number of retries for the agent.
    """
    agent_factory_args: Dict
    max_steps: int
    retries: int

@dataclass
class MiniWobExploreConfig:
    """
    Configuration for the MiniWob++ Explore algorithm.

    Attributes:
        env_names (List[str]): List of MiniWob++ environment names to explore.
        episodes_per_env (int): Number of episodes to run per environment.
        explorer_agent (MiniWobExploreAgentConfig): Configuration for the explorer agent.
        evaluator_agent (MiniWobExploreAgentConfig): Configuration for the evaluator agent.
        exp_dir (str): Directory to save exploration results.
        headless (bool): Whether to run in headless mode.
        slow_mo (int): Slow motion delay in milliseconds.
        viewport_size (Dict): Browser viewport size.
        save_screenshots (bool): Whether to save screenshots during exploration.
        save_traces (bool): Whether to save execution traces.
    """
    env_names: List[str]
    episodes_per_env: int
    explorer_agent: MiniWobExploreAgentConfig
    evaluator_agent: Optional[MiniWobExploreAgentConfig]
    exp_dir: str
    headless: bool = True
    slow_mo: int = 0
    viewport_size: Optional[Dict] = None
    save_screenshots: bool = True
    save_traces: bool = True

    def __post_init__(self):
        if self.viewport_size is None:
            self.viewport_size = {"width": 1280, "height": 720}

class MiniWobExplorer:
    """
    MiniWob++ exploration algorithm that collects trajectories for training.
    
    This explorer runs episodes on MiniWob++ environments, collects successful
    and failed trajectories, and organizes them for downstream training.
    """

    def __init__(self, config: MiniWobExploreConfig):
        """
        Initialize the MiniWob++ explorer.

        Args:
            config (MiniWobExploreConfig): Configuration for the explorer.
        """
        self.config = config
        self.tasks: Dict[str, MiniWobTask] = {}
        
        # Create experiment directory
        os.makedirs(self.config.exp_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.config.exp_dir, "explore_config.yaml")
        oc.save(oc.structured(config), config_path)
        
        logger.info(f"MiniWob++ Explorer initialized with {len(self.config.env_names)} environments")
        logger.info(f"Experiment directory: {self.config.exp_dir}")

    def create_agents(self):
        """Create explorer and evaluator agents."""
        # Create explorer agent
        self.explorer_agent = AgentFactory.create_agent(**self.config.explorer_agent.agent_factory_args)
        self.explorer_agent = wrap_agent_for_callback_protocol(self.explorer_agent)
        
        # Create evaluator agent if specified
        if self.config.evaluator_agent:
            self.evaluator_agent = AgentFactory.create_agent(**self.config.evaluator_agent.agent_factory_args)
            self.evaluator_agent = wrap_agent_for_callback_protocol(self.evaluator_agent)
        else:
            self.evaluator_agent = None
            
        logger.info("Agents created successfully")

    def explore(self):
        """
        Run the exploration process across all specified MiniWob++ environments.
        """
        logger.info("Starting MiniWob++ exploration...")
        
        self.create_agents()
        
        total_episodes = len(self.config.env_names) * self.config.episodes_per_env
        episode_count = 0
        
        for env_name in self.config.env_names:
            logger.info(f"Exploring environment: {env_name}")
            
            # Create task for this environment
            task_dir = os.path.join(self.config.exp_dir, env_name)
            task = MiniWobTask(
                env_name=env_name,
                exp_dir=task_dir,
                config=self.config
            )
            self.tasks[env_name] = task
            
            # Run episodes for this environment
            for episode_idx in range(self.config.episodes_per_env):
                episode_count += 1
                logger.info(f"Running episode {episode_idx + 1}/{self.config.episodes_per_env} "
                           f"for {env_name} (Total: {episode_count}/{total_episodes})")
                
                try:
                    # Run exploration episode
                    result = self._run_exploration_episode(env_name, episode_idx)
                    
                    # Add trajectory to task
                    if result.trajectory:
                        task.add_trajectory(result.trajectory)
                        
                        # Log results
                        status = "SUCCESS" if result.trajectory.success else "FAILURE"
                        logger.info(f"Episode {episode_idx + 1} completed: {status} "
                                   f"(Steps: {len(result.trajectory.steps)}, "
                                   f"Reward: {result.trajectory.final_reward})")
                    else:
                        logger.warning(f"Episode {episode_idx + 1} failed to generate trajectory")
                        
                except Exception as e:
                    logger.error(f"Error in episode {episode_idx + 1} for {env_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
        
        # Generate summary
        self._generate_exploration_summary()
        logger.info("MiniWob++ exploration completed!")

    def _run_exploration_episode(self, env_name: str, episode_idx: int) -> MiniWobEpisodeResult:
        """
        Run a single exploration episode.

        Args:
            env_name (str): Name of the MiniWob++ environment.
            episode_idx (int): Episode index.

        Returns:
            MiniWobEpisodeResult: Result of the episode.
        """
        return run_miniwob_episode(
            env_name=env_name,
            agent=self.explorer_agent,
            max_steps=self.config.explorer_agent.max_steps,
            episode_idx=episode_idx,
            config=self.config,
            save_dir=os.path.join(self.config.exp_dir, env_name, "episodes", f"episode_{episode_idx}")
        )

    def _generate_exploration_summary(self):
        """Generate a summary of the exploration results."""
        summary = {
            "total_environments": len(self.config.env_names),
            "episodes_per_env": self.config.episodes_per_env,
            "total_episodes": len(self.config.env_names) * self.config.episodes_per_env,
            "environments": {}
        }
        
        total_success = 0
        total_episodes = 0
        
        for env_name, task in self.tasks.items():
            env_success = len(task.positive_trajs)
            env_total = len(task.positive_trajs) + len(task.negative_trajs)
            env_success_rate = env_success / env_total if env_total > 0 else 0
            
            summary["environments"][env_name] = {
                "successful_episodes": env_success,
                "failed_episodes": len(task.negative_trajs),
                "total_episodes": env_total,
                "success_rate": env_success_rate
            }
            
            total_success += env_success
            total_episodes += env_total
        
        summary["overall_success_rate"] = total_success / total_episodes if total_episodes > 0 else 0
        summary["total_successful_episodes"] = total_success
        summary["total_failed_episodes"] = total_episodes - total_success
        
        # Save summary
        import json
        summary_path = os.path.join(self.config.exp_dir, "exploration_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Log summary
        logger.info("=== EXPLORATION SUMMARY ===")
        logger.info(f"Total Episodes: {total_episodes}")
        logger.info(f"Successful Episodes: {total_success}")
        logger.info(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
        
        for env_name, env_stats in summary["environments"].items():
            logger.info(f"{env_name}: {env_stats['successful_episodes']}/{env_stats['total_episodes']} "
                       f"({env_stats['success_rate']:.2%})")

def main():
    """Main function for running MiniWob++ exploration from command line."""
    parser = argparse.ArgumentParser(description="Run MiniWob++ exploration")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to exploration configuration file")
    parser.add_argument("--exp_dir", type=str, default=None,
                       help="Override experiment directory")
    parser.add_argument("--episodes_per_env", type=int, default=None,
                       help="Override episodes per environment")
    
    args = parser.parse_args()
    
    # Load configuration
    config_dict = oc.load(args.config)
    
    # Apply overrides
    if args.exp_dir:
        config_dict.exp_dir = args.exp_dir
    if args.episodes_per_env:
        config_dict.episodes_per_env = args.episodes_per_env
    
    # Create config object
    config = oc.structured(MiniWobExploreConfig, config_dict)
    
    # Run exploration
    explorer = MiniWobExplorer(config)
    explorer.explore()

if __name__ == "__main__":
    main()