#!/usr/bin/env python3
"""
Test trajectory attribute fix.
"""

import sys
import os
sys.path.append('/workspace/Go-Browse')

from webexp.explore.core.trajectory import Trajectory

def test_trajectory_attributes():
    """Test that Trajectory has the correct attributes."""
    
    # Create a test trajectory
    trajectory = Trajectory.from_goal("Test goal")
    
    # Check that it has the correct attributes
    print(f"Trajectory attributes: {dir(trajectory)}")
    
    # Check specific attributes
    assert hasattr(trajectory, 'reward'), "Trajectory should have 'reward' attribute"
    assert not hasattr(trajectory, 'final_reward'), "Trajectory should NOT have 'final_reward' attribute"
    
    print(f"✅ Trajectory.reward: {trajectory.reward}")
    print(f"✅ Trajectory has correct attributes")
    
    # Test setting reward
    trajectory.reward = 1.0
    print(f"✅ Set reward to: {trajectory.reward}")
    
    print("✅ Trajectory attribute test passed!")

if __name__ == "__main__":
    test_trajectory_attributes()