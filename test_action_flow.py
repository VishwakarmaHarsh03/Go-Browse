#!/usr/bin/env python3
"""
Test the complete action flow from raw action to MiniWob++ action.
"""

import json
import re
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/workspace/Go-Browse')

from webexp.explore.algorithms.miniwob_episode import _convert_to_miniwob_action

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

def test_action_flow():
    """Test the complete action flow."""
    
    # Test the raw action from the log
    raw_action = '```json\n{"thought": "The task is to click the button. The button element has ref: 4.", "action": "click(4)"}\n```'
    
    print(f"Raw action: {raw_action}")
    
    # Parse the action
    parsed_action = parse_action(raw_action)
    print(f"Parsed action: {parsed_action}")
    
    # Convert to MiniWob++ format
    miniwob_action = _convert_to_miniwob_action(None, parsed_action)
    print(f"MiniWob++ action: {miniwob_action}")
    
    # Verify the format
    assert 'action_type' in miniwob_action
    assert isinstance(miniwob_action['action_type'], str)
    print(f"✅ Action type is string: {miniwob_action['action_type']}")
    
    print("✅ Action flow test passed!")

if __name__ == "__main__":
    test_action_flow()