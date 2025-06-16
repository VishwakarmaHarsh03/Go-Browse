#!/usr/bin/env python3
"""
Test script to verify action conversion works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from webexp.explore.algorithms.miniwob_episode import _convert_to_miniwob_action

def test_action_conversion():
    """Test various action conversions."""
    
    print("Testing action conversion...")
    
    # Test click with element reference
    click_action = {'type': 'click', 'ref': '4'}
    result = _convert_to_miniwob_action(None, click_action)
    print(f"Click action: {click_action} -> {result}")
    
    # Test click with coordinates
    click_coord_action = {'type': 'click', 'coordinate': [100, 200]}
    result = _convert_to_miniwob_action(None, click_coord_action)
    print(f"Click coord action: {click_coord_action} -> {result}")
    
    # Test type with element reference
    type_action = {'type': 'type', 'ref': '5', 'text': 'hello world'}
    result = _convert_to_miniwob_action(None, type_action)
    print(f"Type action: {type_action} -> {result}")
    
    # Test type with text only
    type_text_action = {'type': 'type', 'text': 'hello world'}
    result = _convert_to_miniwob_action(None, type_text_action)
    print(f"Type text action: {type_text_action} -> {result}")
    
    # Test key action
    key_action = {'type': 'key', 'key': 'Enter'}
    result = _convert_to_miniwob_action(None, key_action)
    print(f"Key action: {key_action} -> {result}")
    
    # Test scroll action
    scroll_action = {'type': 'scroll', 'coordinate': [100, 100], 'direction': 'down'}
    result = _convert_to_miniwob_action(None, scroll_action)
    print(f"Scroll action: {scroll_action} -> {result}")
    
    # Test invalid action
    invalid_action = {'type': 'unknown'}
    result = _convert_to_miniwob_action(None, invalid_action)
    print(f"Invalid action: {invalid_action} -> {result}")
    
    print("Action conversion test completed successfully!")

if __name__ == "__main__":
    test_action_conversion()