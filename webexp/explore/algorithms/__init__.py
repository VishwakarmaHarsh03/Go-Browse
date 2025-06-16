"""
Exploration algorithms for different environments.
"""

# Import WebArena components only if available
try:
    from .web_explore import WebExploreConfig, WebExploreAgentConfig
    _web_explore_available = True
except ImportError:
    _web_explore_available = False

# MiniWob++ components (should always be available)
from .miniwob_explore import MiniWobExplorer, MiniWobExploreConfig, MiniWobExploreAgentConfig
from .miniwob_task import MiniWobTask
from .miniwob_episode import run_miniwob_episode, MiniWobEpisodeResult, run_miniwob_evaluation_episode

__all__ = [
    "MiniWobExplorer",
    "MiniWobExploreConfig",
    "MiniWobExploreAgentConfig",
    "MiniWobTask",
    "run_miniwob_episode",
    "MiniWobEpisodeResult",
    "run_miniwob_evaluation_episode"
]

# Add WebArena components to __all__ if available
if _web_explore_available:
    __all__.extend([
        "WebExploreConfig", 
        "WebExploreAgentConfig"
    ])