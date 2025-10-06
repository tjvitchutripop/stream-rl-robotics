#!/usr/bin/env python3

"""
Test script to verify the accumulative damage logic
"""

def get_accumulative_damage_types(damage_type, damage_start_step, damage_steps, t):
    """Get all damage types that should be active at timestep t (accumulative)."""
    if t < damage_start_step:
        return []
    
    active_damages = []
    steps_since_damage_start = t - damage_start_step
    total_damage_duration = len(damage_type) * damage_steps
    
    # Check if we're still within the total damage sequence
    if steps_since_damage_start >= total_damage_duration:
        # All damages should remain active even after the sequence ends (true accumulation)
        return damage_type[:]
    
    # For each damage type, check if it should be active
    for i, damage in enumerate(damage_type):
        damage_start = i * damage_steps
        # Once a damage starts, it stays active (accumulative)
        if steps_since_damage_start >= damage_start:
            active_damages.append(damage)
    
    return active_damages

def get_newly_introduced_damage(damage_type, damage_start_step, damage_steps, t):
    """Get the damage type that was just introduced at this timestep."""
    if t < damage_start_step:
        return None
    
    steps_since_damage_start = t - damage_start_step
    
    # Check if we're exactly at the start of a new damage cycle
    for i, damage in enumerate(damage_type):
        damage_start = i * damage_steps
        if steps_since_damage_start == damage_start:
            return damage
    
    return None

def test_damage_logic():
    damage_types = ['stuck_joint', 'slippery_floor_easy', 'goal_shift_easy']
    damage_start_step = 500_000
    damage_steps = 1_000_000  # This matches the file default
    
    # Test key timesteps
    test_steps = [
        0,                    # Before damage starts
        499_999,              # Just before damage starts
        500_000,              # First damage starts (stuck_joint)
        500_001,              # Just after first damage starts
        1_499_999,            # Just before second damage
        1_500_000,            # Second damage starts (slippery_floor_easy)
        1_500_001,            # Just after second damage starts
        2_499_999,            # Just before third damage
        2_500_000,            # Third damage starts (goal_shift_easy)
        2_500_001,            # Just after third damage starts
        3_500_000,            # All damages should still be active (accumulative)
        4_000_000,            # Well past sequence, all should still be active
    ]
    
    print("Testing accumulative damage logic:")
    print(f"Damage types: {damage_types}")
    print(f"Damage start step: {damage_start_step}")
    print(f"Damage steps: {damage_steps}")
    print()
    
    for step in test_steps:
        active = get_accumulative_damage_types(damage_types, damage_start_step, damage_steps, step)
        newly_introduced = get_newly_introduced_damage(damage_types, damage_start_step, damage_steps, step)
        
        print(f"Step {step:,}:")
        print(f"  Active damages: {active}")
        print(f"  Newly introduced: {newly_introduced}")
        print()

if __name__ == "__main__":
    test_damage_logic()
