#!/usr/bin/env python3
"""
Test script to verify action format works correctly.
"""

from miniwob.action import ActionTypes

def test_action_format():
    """Test action format for MiniWob++."""
    
    print("Testing action format...")
    
    # Test the format that should work
    action = {'action_type': ActionTypes.CLICK_ELEMENT.value, 'element': 4}
    print(f"Correct format: {action}")
    print(f"action_type value: {action['action_type']}")
    print(f"action_type type: {type(action['action_type'])}")
    
    # Test what we were doing wrong
    wrong_action = {'action_type': ActionTypes.CLICK_ELEMENT, 'element': 4}
    print(f"Wrong format: {wrong_action}")
    print(f"action_type value: {wrong_action['action_type']}")
    print(f"action_type type: {type(wrong_action['action_type'])}")
    
    print("Action format test completed!")

if __name__ == "__main__":
    test_action_format()