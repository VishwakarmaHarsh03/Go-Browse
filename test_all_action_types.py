#!/usr/bin/env python3
"""
Test all action types with complete field format.
"""

from miniwob.action import ActionTypes

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
    elif action_type == 'type':
        # Check if it's field-based or text-based
        if 'ref' in action_dict:
            # Field-based type
            element_ref = action_dict['ref']
            text = action_dict.get('text', '')
            try:
                element_id = int(element_ref)
                return create_complete_action(
                    action_type_index=get_action_index(ActionTypes.TYPE_FIELD),
                    text=text,
                    ref=element_id
                )
            except ValueError:
                # If ref is not a number, use text-based type
                return create_complete_action(
                    action_type_index=get_action_index(ActionTypes.TYPE_TEXT),
                    text=text
                )
        else:
            # Text-based type
            text = action_dict.get('text', '')
            return create_complete_action(
                action_type_index=get_action_index(ActionTypes.TYPE_TEXT),
                text=text
            )
    elif action_type == 'key':
        key = action_dict.get('key', 'Enter')
        return create_complete_action(
            action_type_index=get_action_index(ActionTypes.PRESS_KEY),
            key_comb=key
        )
    elif action_type == 'scroll':
        coordinate = action_dict.get('coordinate', [0, 0])
        direction = action_dict.get('direction', 'down')
        if direction == 'down':
            return create_complete_action(
                action_type_index=get_action_index(ActionTypes.SCROLL_DOWN_COORDS),
                coordinate=coordinate
            )
        else:
            return create_complete_action(
                action_type_index=get_action_index(ActionTypes.SCROLL_UP_COORDS),
                coordinate=coordinate
            )
    else:
        return create_complete_action(get_action_index(ActionTypes.NONE))

def test_all_action_types():
    """Test all action types to ensure complete field format."""
    
    test_cases = [
        # Click element
        {'type': 'click', 'ref': 4},
        # Click coordinates
        {'type': 'click', 'coordinate': [100, 200]},
        # Type in field
        {'type': 'type', 'ref': 2, 'text': 'hello world'},
        # Type text
        {'type': 'type', 'text': 'hello world'},
        # Press key
        {'type': 'key', 'key': 'Enter'},
        # Scroll down
        {'type': 'scroll', 'coordinate': [50, 50], 'direction': 'down'},
        # Scroll up
        {'type': 'scroll', 'coordinate': [50, 50], 'direction': 'up'},
        # None action
        {'type': 'none'}
    ]
    
    required_fields = ['action_type', 'coordinate', 'text', 'key_comb', 'ref']
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case} ---")
        
        miniwob_action = _convert_to_miniwob_action(None, test_case)
        print(f"Result: {miniwob_action}")
        
        # Verify all required fields are present
        for field in required_fields:
            assert field in miniwob_action, f"Missing required field: {field}"
        
        # Verify field types
        assert isinstance(miniwob_action['action_type'], int)
        assert isinstance(miniwob_action['coordinate'], list)
        assert len(miniwob_action['coordinate']) == 2
        assert isinstance(miniwob_action['text'], str)
        assert isinstance(miniwob_action['key_comb'], str)
        assert isinstance(miniwob_action['ref'], int)
        
        print(f"✅ All fields present and correct types")
    
    print("\n✅ All action types test passed!")

if __name__ == "__main__":
    test_all_action_types()