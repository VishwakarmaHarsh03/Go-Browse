import gymnasium
import miniwob
from ..agents.base_agent import AgentFactory
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
import argparse
import os
import json
import time
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class RunMiniWobConfig:
    """
    Configuration for running an agent on MiniWob++ benchmark.
    
    Attributes:
        agent_factory_args (dict): Arguments for the agent factory.
        exp_dir (str): Directory to save the experiment results.
        environments (list): List of MiniWob++ environment names to test.
        num_episodes (int): Number of episodes to run per environment.
        max_steps (int): Maximum steps per episode.
        timeout (int): Timeout in seconds per episode.
    """
    agent_factory_args: dict
    exp_dir: str
    environments: List[str]
    num_episodes: int = 5
    max_steps: int = 50
    timeout: int = 60


class MiniWobBenchmark:
    """Benchmark runner for MiniWob++ environments."""
    
    def __init__(self, config: RunMiniWobConfig):
        self.config = config
        self.agent = AgentFactory.create_agent(**config.agent_factory_args)
        
        # Register MiniWob++ environments
        gymnasium.register_envs(miniwob)
        
        # Create experiment directory
        os.makedirs(config.exp_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.exp_dir, 'benchmark.log')),
                logging.StreamHandler()
            ]
        )
        
    def run_episode(self, env_name: str, episode_idx: int) -> Dict[str, Any]:
        """Run a single episode on the given environment."""
        
        env = gymnasium.make(f'miniwob/{env_name}-v1', render_mode=None)
        
        try:
            # Reset agent and environment
            self.agent.reset()
            obs, info = env.reset()
            
            episode_data = {
                'env_name': env_name,
                'episode_idx': episode_idx,
                'steps': [],
                'total_reward': 0.0,
                'success': False,
                'terminated': False,
                'truncated': False,
                'error': None,
                'start_time': time.time()
            }
            
            for step_idx in range(self.config.max_steps):
                try:
                    # Preprocess observation for the agent
                    processed_obs = self.agent.obs_preprocessor(obs)
                    
                    # Get action from agent
                    raw_action, action_info = self.agent.get_action(processed_obs)
                    
                    # Process action for the environment
                    action_dict = self.agent.action_processor(raw_action)
                    
                    # Convert action dict to MiniWob++ action
                    miniwob_action = self._convert_to_miniwob_action(env, action_dict)
                    
                    # Execute action
                    next_obs, reward, terminated, truncated, step_info = env.step(miniwob_action)
                    
                    # Record step data
                    step_data = {
                        'step_idx': step_idx,
                        'raw_action': raw_action,
                        'processed_action': str(processed_action),
                        'reward': reward,
                        'terminated': terminated,
                        'truncated': truncated,
                        'action_info': action_info,
                        'step_info': step_info
                    }
                    episode_data['steps'].append(step_data)
                    episode_data['total_reward'] += reward
                    
                    # Check if episode is done
                    if terminated or truncated:
                        episode_data['success'] = terminated and reward > 0
                        episode_data['terminated'] = terminated
                        episode_data['truncated'] = truncated
                        break
                        
                    obs = next_obs
                    
                except Exception as e:
                    logger.error(f"Error in step {step_idx} of episode {episode_idx} for {env_name}: {str(e)}")
                    episode_data['error'] = str(e)
                    break
                    
            episode_data['end_time'] = time.time()
            episode_data['duration'] = episode_data['end_time'] - episode_data['start_time']
            
            return episode_data
            
        finally:
            env.close()
    
    def _convert_to_miniwob_action(self, env, action_dict):
        """Convert action dictionary to MiniWob++ action object."""
        from miniwob.action import ActionTypes
        
        if not isinstance(action_dict, dict):
            return ActionTypes.NONE
            
        action_type = action_dict.get('type', 'none')
        
        if action_type == 'click':
            ref = action_dict.get('ref')
            if ref:
                return env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref)
        
        elif action_type == 'type':
            text = action_dict.get('text', '')
            return env.unwrapped.create_action(ActionTypes.TYPE, text=text)
        
        elif action_type == 'key':
            key = action_dict.get('key', '')
            return env.unwrapped.create_action(ActionTypes.KEY, key=key)
        
        elif action_type == 'scroll':
            direction = action_dict.get('direction', 'down')
            if direction.lower() == 'up':
                return env.unwrapped.create_action(ActionTypes.SCROLL, coordinate=[0, -100])
            elif direction.lower() == 'down':
                return env.unwrapped.create_action(ActionTypes.SCROLL, coordinate=[0, 100])
            elif direction.lower() == 'left':
                return env.unwrapped.create_action(ActionTypes.SCROLL, coordinate=[-100, 0])
            elif direction.lower() == 'right':
                return env.unwrapped.create_action(ActionTypes.SCROLL, coordinate=[100, 0])
        
        # Default to no action
        return ActionTypes.NONE
    
    def run_environment(self, env_name: str) -> Dict[str, Any]:
        """Run multiple episodes on a single environment."""
        
        logger.info(f"Starting benchmark for environment: {env_name}")
        
        env_results = {
            'env_name': env_name,
            'episodes': [],
            'summary': {
                'total_episodes': self.config.num_episodes,
                'successful_episodes': 0,
                'average_reward': 0.0,
                'average_duration': 0.0,
                'success_rate': 0.0
            }
        }
        
        for episode_idx in range(self.config.num_episodes):
            logger.info(f"Running episode {episode_idx + 1}/{self.config.num_episodes} for {env_name}")
            
            episode_data = self.run_episode(env_name, episode_idx)
            env_results['episodes'].append(episode_data)
            
            if episode_data['success']:
                env_results['summary']['successful_episodes'] += 1
                
        # Calculate summary statistics
        total_reward = sum(ep['total_reward'] for ep in env_results['episodes'])
        total_duration = sum(ep['duration'] for ep in env_results['episodes'])
        
        env_results['summary']['average_reward'] = total_reward / self.config.num_episodes
        env_results['summary']['average_duration'] = total_duration / self.config.num_episodes
        env_results['summary']['success_rate'] = env_results['summary']['successful_episodes'] / self.config.num_episodes
        
        logger.info(f"Completed {env_name}: Success rate = {env_results['summary']['success_rate']:.2%}")
        
        return env_results
    
    def run(self) -> Dict[str, Any]:
        """Run the full benchmark across all environments."""
        
        logger.info(f"Starting MiniWob++ benchmark with {len(self.config.environments)} environments")
        
        benchmark_results = {
            'config': {
                'agent_config': self.agent.get_config(),
                'num_episodes': self.config.num_episodes,
                'max_steps': self.config.max_steps,
                'timeout': self.config.timeout,
                'environments': self.config.environments
            },
            'results': [],
            'overall_summary': {
                'total_environments': len(self.config.environments),
                'total_episodes': len(self.config.environments) * self.config.num_episodes,
                'successful_episodes': 0,
                'average_success_rate': 0.0,
                'environment_success_rates': {}
            }
        }
        
        for env_name in self.config.environments:
            env_results = self.run_environment(env_name)
            benchmark_results['results'].append(env_results)
            
            # Update overall summary
            benchmark_results['overall_summary']['successful_episodes'] += env_results['summary']['successful_episodes']
            benchmark_results['overall_summary']['environment_success_rates'][env_name] = env_results['summary']['success_rate']
        
        # Calculate overall success rate
        benchmark_results['overall_summary']['average_success_rate'] = (
            benchmark_results['overall_summary']['successful_episodes'] / 
            benchmark_results['overall_summary']['total_episodes']
        )
        
        # Save results
        results_file = os.path.join(self.config.exp_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
            
        logger.info(f"Benchmark completed. Overall success rate: {benchmark_results['overall_summary']['average_success_rate']:.2%}")
        logger.info(f"Results saved to: {results_file}")
        
        return benchmark_results


def run():
    parser = argparse.ArgumentParser(description="Run MiniWob++ benchmark.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config: RunMiniWobConfig = oc.load(args.config)
    oc.resolve(config)
    
    benchmark = MiniWobBenchmark(config)
    results = benchmark.run()
    
    return results


if __name__ == "__main__":
    run()