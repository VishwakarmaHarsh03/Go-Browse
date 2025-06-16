#!/usr/bin/env python3
"""
Test to find the correct action type indices for MiniWob++.
"""

from miniwob.action import ActionTypes

def test_action_indices():
    """Test action type indices."""
    
    print("ActionTypes enum values and indices:")
    action_types_list = list(ActionTypes)
    
    for i, action_type in enumerate(action_types_list):
        print(f"{i}: {action_type} = '{action_type.value}'")
    
    # Find specific indices we need
    click_element_index = None
    click_coords_index = None
    type_text_index = None
    type_field_index = None
    press_key_index = None
    
    for i, action_type in enumerate(action_types_list):
        if action_type == ActionTypes.CLICK_ELEMENT:
            click_element_index = i
        elif action_type == ActionTypes.CLICK_COORDS:
            click_coords_index = i
        elif action_type == ActionTypes.TYPE_TEXT:
            type_text_index = i
        elif action_type == ActionTypes.TYPE_FIELD:
            type_field_index = i
        elif action_type == ActionTypes.PRESS_KEY:
            press_key_index = i
    
    print(f"\nKey action indices:")
    print(f"CLICK_ELEMENT: {click_element_index}")
    print(f"CLICK_COORDS: {click_coords_index}")
    print(f"TYPE_TEXT: {type_text_index}")
    print(f"TYPE_FIELD: {type_field_index}")
    print(f"PRESS_KEY: {press_key_index}")

if __name__ == "__main__":
    test_action_indices()