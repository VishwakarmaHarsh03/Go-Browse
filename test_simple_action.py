#!/usr/bin/env python3
"""
Simple test of action conversion without full imports.
"""

import json
import re
from miniwob.action import ActionTypes

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

def _convert_to_miniwob_action(env, action_dict):
    """Convert action dictionary to MiniWob++ action object."""
    
    if not isinstance(action_dict, dict):
        return {'action_type': ActionTypes.NONE.value}
        
    action_type = action_dict.get('type', 'none')
    
    if action_type == 'click':
        # Check if it's element-based or coordinate-based
        if 'ref' in action_dict:
            # Element-based click
            element_ref = action_dict['ref']
            try:
                element_id = int(element_ref)
                return {'action_type': ActionTypes.CLICK_ELEMENT.value, 'element': element_id}
            except ValueError:
                # If ref is not a number, treat as coordinate [0, 0]
                return {'action_type': ActionTypes.CLICK_COORDS.value, 'coord': [0, 0]}
        else:
            # Coordinate-based click
            coordinate = action_dict.get('coordinate', [0, 0])
            return {'action_type': ActionTypes.CLICK_COORDS.value, 'coord': coordinate}
    else:
        return {'action_type': ActionTypes.NONE.value}

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
    assert miniwob_action['action_type'] == 'CLICK_ELEMENT'
    assert miniwob_action['element'] == 4
    print(f"✅ Action type is string: {miniwob_action['action_type']}")
    print(f"✅ Element ID is correct: {miniwob_action['element']}")
    
    print("✅ Action flow test passed!")

if __name__ == "__main__":
    test_action_flow()