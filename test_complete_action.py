#!/usr/bin/env python3
"""
Test complete action format with all required fields.
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
    """Convert action dictionary to MiniWob++ action object with all required fields."""
    
    # Get the index of each action type in the ActionTypes enum
    action_types_list = list(ActionTypes)
    
    def get_action_index(action_type_enum):
        return action_types_list.index(action_type_enum)
    
    def create_complete_action(action_type_index, coordinate=None, text="", key_comb="", ref=None):
        """Create a complete action dictionary with all required fields."""
        return {
            'action_type': action_type_index,
            'coordinate': coordinate if coordinate is not None else [0, 0],
            'text': text,
            'key_comb': key_comb,
            'ref': ref if ref is not None else 0
        }
    
    if not isinstance(action_dict, dict):
        return create_complete_action(get_action_index(ActionTypes.NONE))
        
    action_type = action_dict.get('type', 'none')
    
    if action_type == 'click':
        # Check if it's element-based or coordinate-based
        if 'ref' in action_dict:
            # Element-based click
            element_ref = action_dict['ref']
            try:
                element_id = int(element_ref)
                return create_complete_action(
                    action_type_index=get_action_index(ActionTypes.CLICK_ELEMENT),
                    coordinate=[0, 0],  # Will be filled by environment
                    ref=element_id
                )
            except ValueError:
                # If ref is not a number, treat as coordinate [0, 0]
                return create_complete_action(
                    action_type_index=get_action_index(ActionTypes.CLICK_COORDS),
                    coordinate=[0, 0]
                )
        else:
            # Coordinate-based click
            coordinate = action_dict.get('coordinate', [0, 0])
            return create_complete_action(
                action_type_index=get_action_index(ActionTypes.CLICK_COORDS),
                coordinate=coordinate
            )
    else:
        return create_complete_action(get_action_index(ActionTypes.NONE))

def test_complete_action_format():
    """Test the complete action format with all required fields."""
    
    # Test the raw action from the log
    raw_action = '```json\n{"thought": "The task is to click the button. The button element has ref: 4.", "action": "click(4)"}\n```'
    
    print(f"Raw action: {raw_action}")
    
    # Parse the action
    parsed_action = parse_action(raw_action)
    print(f"Parsed action: {parsed_action}")
    
    # Convert to MiniWob++ format
    miniwob_action = _convert_to_miniwob_action(None, parsed_action)
    print(f"Complete MiniWob++ action: {miniwob_action}")
    
    # Verify all required fields are present
    required_fields = ['action_type', 'coordinate', 'text', 'key_comb', 'ref']
    for field in required_fields:
        assert field in miniwob_action, f"Missing required field: {field}"
        print(f"✅ {field}: {miniwob_action[field]}")
    
    # Verify the specific values
    assert isinstance(miniwob_action['action_type'], int)
    assert miniwob_action['action_type'] == 8  # CLICK_ELEMENT index
    assert isinstance(miniwob_action['coordinate'], list)
    assert len(miniwob_action['coordinate']) == 2
    assert isinstance(miniwob_action['text'], str)
    assert isinstance(miniwob_action['key_comb'], str)
    assert isinstance(miniwob_action['ref'], int)
    assert miniwob_action['ref'] == 4
    
    print("✅ Complete action format test passed!")
    print(f"Final action: {miniwob_action}")

if __name__ == "__main__":
    test_complete_action_format()