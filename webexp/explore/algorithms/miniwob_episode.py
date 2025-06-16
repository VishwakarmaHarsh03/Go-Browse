"""
MiniWob++ Episode Management

This module provides episode running capabilities for MiniWob++ environments
during exploration and data collection.
"""

from ..core.trajectory import Trajectory, TrajectoryStep
from ...agents.base_agent import Agent
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import json
import logging
import os
import time
import traceback
import re
logger = logging.getLogger(__name__)

@dataclass
class MiniWobEpisodeResult:
    """Result of running a MiniWob++ episode."""
    success: bool
    trajectory: Optional[Trajectory]
    final_reward: float
    steps_taken: int
    error: Optional[str] = None
    episode_info: Optional[Dict] = None

def run_miniwob_episode(
    env_name: str,
    agent: Agent,
    max_steps: int,
    episode_idx: int,
    config: Any,
    save_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> MiniWobEpisodeResult:
    """
    Run a single MiniWob++ episode for exploration.
    
    Args:
        env_name (str): Name of the MiniWob++ environment.
        agent (Agent): Agent to run the episode.
        max_steps (int): Maximum number of steps.
        episode_idx (int): Episode index for logging.
        config (Any): Configuration object.
        save_dir (Optional[str]): Directory to save episode data.
        seed (Optional[int]): Random seed for reproducibility.
        
    Returns:
        MiniWobEpisodeResult: Result of the episode.
    """
    trajectory_steps = []
    episode_info = {
        "env_name": env_name,
        "episode_idx": episode_idx,
        "start_time": time.time(),
        "max_steps": max_steps,
        "seed": seed
    }
    
    try:
        # Import MiniWob++ environment
        import miniwob
        from miniwob.action import ActionTypes
        
        # Create environment
        env = gym.make(f"miniwob/{env_name}-v1")
        
        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset()
        
        # Get initial observation
        obs, info = env.reset()
        
        # Initialize agent
        agent.reset()
        
        # Process initial observation
        processed_obs = agent.obs_preprocessor(obs)
        
        total_reward = 0.0
        step_count = 0
        
        for step_idx in range(max_steps):
            step_start_time = time.time()
            
        
            # Get action from agent
            raw_action, action_info = agent.get_action(processed_obs)
            def parse_action(raw_action_str):
                # Remove markdown code block markers if present
                raw_action_str = raw_action_str.strip()
                if raw_action_str.startswith("```json"):
                    raw_action_str = raw_action_str[len("```json"):].strip()
                if raw_action_str.startswith("```"):
                    raw_action_str = raw_action_str[len("```"):].strip()
                if raw_action_str.endswith("```"):
                    raw_action_str = raw_action_str[:-3].strip()
                # Parse JSON
                action_obj = json.loads(raw_action_str)
                action_str = action_obj["action"]
                # Accepts click(4), click('4'), or click("4")
                match = re.match(r"(\w+)\(['\"]?(\d+)['\"]?\)", action_str)
                if match:
                    return {"type": match.group(1), "ref": int(match.group(2))}
                raise ValueError(f"Invalid action format: {action_str}")



            # Usage:
        
            parsed_action = parse_action(raw_action)
            logger.info(f"Parsed action: {parsed_action}")
            # Convert to MiniWob++ action format
            miniwob_action = _convert_to_miniwob_action(env, parsed_action)
            logger.info(f"MiniWob++ action: {miniwob_action}")
            # Execute action
            next_obs, reward, terminated, truncated, step_info = env.step(miniwob_action)
            
            # Create trajectory step
            traj_step = TrajectoryStep(
                action=raw_action,
                parsed_action=str(parsed_action),
                thought=action_info.get('thought', '') if action_info else '',
                observation=processed_obs,
                misc={
                    "step_idx": step_idx,
                    "parsed_action": parsed_action,
                    "miniwob_action": str(miniwob_action),
                    "step_info": step_info,
                    "action_info": action_info,
                    "step_time": time.time() - step_start_time,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated
                }
            )
            trajectory_steps.append(traj_step)
            
            total_reward += reward
            step_count += 1
            
            # Check if episode is done
            if terminated or truncated:
                episode_info["terminated"] = terminated
                episode_info["truncated"] = truncated
                episode_info["success"] = terminated and reward > 0
                break
            
            # Update observation for next step
            processed_obs = agent.obs_preprocessor(next_obs)
                
           
        
        # Finalize episode info
        episode_info["end_time"] = time.time()
        episode_info["duration"] = episode_info["end_time"] - episode_info["start_time"]
        episode_info["total_reward"] = total_reward
        episode_info["steps_taken"] = step_count
        episode_info["final_reward"] = total_reward
        
        # Determine success
        success = episode_info.get("success", False)
        
        # Create trajectory
        trajectory = Trajectory(
            steps=trajectory_steps,
            final_state=trajectory_steps[-1] if trajectory_steps else None,
            goal=env_name,  # Use environment name as goal
            reward=total_reward,
            success=success,
            response="",  # Will be filled by extract_response if needed
            agent_info={
                "agent_type": type(agent).__name__,
                "config": {
                    "max_steps": max_steps,
                    "headless": getattr(config, 'headless', True)
                }
            },
            misc={
                "env_name": env_name,
                "episode_idx": episode_idx,
                "duration": episode_info["duration"],
                "total_steps": step_count
            }
        )
        
        # Save episode data if requested
        if save_dir:
            _save_episode_data(save_dir, trajectory, episode_info, config)
        
        env.close()
        
        return MiniWobEpisodeResult(
            success=success,
            trajectory=trajectory,
            final_reward=total_reward,
            steps_taken=step_count,
            episode_info=episode_info
        )
        
    except Exception as e:
        error_msg = f"Failed to run episode {episode_idx} for {env_name}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return MiniWobEpisodeResult(
            success=False,
            trajectory=None,
            final_reward=0.0,
            steps_taken=0,
            error=error_msg,
            episode_info=episode_info
        )

def _convert_to_miniwob_action(env, action_dict):
    """Convert action dictionary to MiniWob++ action object."""
    from miniwob.action import ActionTypes
    
    # Get the index of each action type in the ActionTypes enum
    action_types_list = list(ActionTypes)
    
    def get_action_index(action_type_enum):
        return action_types_list.index(action_type_enum)
    
    if not isinstance(action_dict, dict):
        return {'action_type': get_action_index(ActionTypes.NONE)}
        
    action_type = action_dict.get('type', 'none')
    
    if action_type == 'click':
        # Check if it's element-based or coordinate-based
        if 'ref' in action_dict:
            # Element-based click
            element_ref = action_dict['ref']
            try:
                element_id = int(element_ref)
                return {'action_type': get_action_index(ActionTypes.CLICK_ELEMENT), 'element': element_id}
            except ValueError:
                # If ref is not a number, treat as coordinate [0, 0]
                return {'action_type': get_action_index(ActionTypes.CLICK_COORDS), 'coord': [0, 0]}
        else:
            # Coordinate-based click
            coordinate = action_dict.get('coordinate', [0, 0])
            return {'action_type': get_action_index(ActionTypes.CLICK_COORDS), 'coord': coordinate}
    elif action_type == 'type':
        # Check if it's field-based or text-based
        if 'ref' in action_dict:
            # Field-based type
            element_ref = action_dict['ref']
            text = action_dict.get('text', '')
            try:
                element_id = int(element_ref)
                return {'action_type': get_action_index(ActionTypes.TYPE_FIELD), 'element': element_id, 'text': text}
            except ValueError:
                # If ref is not a number, use text-based type
                return {'action_type': get_action_index(ActionTypes.TYPE_TEXT), 'text': text}
        else:
            # Text-based type
            text = action_dict.get('text', '')
            return {'action_type': get_action_index(ActionTypes.TYPE_TEXT), 'text': text}
    elif action_type == 'key':
        key = action_dict.get('key', 'Enter')
        return {'action_type': get_action_index(ActionTypes.PRESS_KEY), 'key': key}
    elif action_type == 'scroll':
        coordinate = action_dict.get('coordinate', [0, 0])
        direction = action_dict.get('direction', 'down')
        if direction == 'down':
            return {'action_type': get_action_index(ActionTypes.SCROLL_DOWN_COORDS), 'coord': coordinate}
        else:
            return {'action_type': get_action_index(ActionTypes.SCROLL_UP_COORDS), 'coord': coordinate}
    elif action_type == 'drag':
        start_coord = action_dict.get('startCoordinate', [0, 0])
        end_coord = action_dict.get('endCoordinate', [0, 0])
        # MiniWob++ doesn't have a direct drag action, use mousedown -> move -> mouseup
        return {'action_type': get_action_index(ActionTypes.MOUSEDOWN_COORDS), 'coord': start_coord}
    else:
        return {'action_type': get_action_index(ActionTypes.NONE)}

def _save_episode_data(save_dir: str, trajectory: Trajectory, episode_info: Dict, config: Any):
    """Save episode data to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save trajectory
    trajectory.save(save_dir)
    
    # Save episode info
    episode_info_path = os.path.join(save_dir, "episode_info.json")
    with open(episode_info_path, "w") as f:
        json.dump(episode_info, f, indent=2, default=str)
    
    # Save screenshots if enabled
    if getattr(config, 'save_screenshots', True):
        _save_screenshots(save_dir, trajectory)

def _save_screenshots(save_dir: str, trajectory: Trajectory):
    """Save screenshots from trajectory steps."""
    screenshots_dir = os.path.join(save_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    for i, step in enumerate(trajectory.steps):
        # Extract screenshot from observation if available
        obs = step.observation
        if isinstance(obs, dict) and 'screenshot' in obs:
            screenshot_path = os.path.join(screenshots_dir, f"step_{i:03d}.png")
            try:
                import base64
                from PIL import Image
                import io
                
                # Decode base64 screenshot
                screenshot_data = base64.b64decode(obs['screenshot'])
                image = Image.open(io.BytesIO(screenshot_data))
                image.save(screenshot_path)
            except Exception as e:
                logger.warning(f"Failed to save screenshot for step {i}: {e}")

def run_miniwob_evaluation_episode(
    env_name: str,
    agent: Agent,
    max_steps: int,
    episode_idx: int,
    reference_trajectory: Optional[Trajectory] = None,
    config: Any = None
) -> MiniWobEpisodeResult:
    """
    Run a MiniWob++ episode for evaluation purposes.
    
    This function is similar to run_miniwob_episode but includes additional
    evaluation-specific features like trajectory comparison.
    
    Args:
        env_name (str): Name of the MiniWob++ environment.
        agent (Agent): Agent to evaluate.
        max_steps (int): Maximum number of steps.
        episode_idx (int): Episode index.
        reference_trajectory (Optional[Trajectory]): Reference trajectory for comparison.
        config (Any): Configuration object.
        
    Returns:
        MiniWobEpisodeResult: Result of the evaluation episode.
    """
    # For now, use the same implementation as exploration
    # Can be extended with evaluation-specific features later
    result = run_miniwob_episode(
        env_name=env_name,
        agent=agent,
        max_steps=max_steps,
        episode_idx=episode_idx,
        config=config
    )
    
    # Add evaluation-specific information
    if reference_trajectory and result.trajectory:
        result.episode_info["evaluation"] = {
            "reference_steps": len(reference_trajectory.steps),
            "actual_steps": len(result.trajectory.steps),
            "reference_success": reference_trajectory.success,
            "actual_success": result.trajectory.success
        }
    
    return result