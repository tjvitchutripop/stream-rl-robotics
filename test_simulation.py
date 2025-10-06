#!/usr/bin/env python3

"""
Simulate the damage initialization process
"""

class MockEnvironment:
    def __init__(self):
        self.friction_changed = False
        self.goal_offset_changed = False
        
    def change_friction(self, x, y):
        print(f"Environment: Changed friction to ({x}, {y})")
        self.friction_changed = True
        
    def set_goal_offset(self, x, y):
        print(f"Environment: Set goal offset to ({x}, {y})")
        self.goal_offset_changed = True

class MockRunner:
    def __init__(self):
        self.damage_type = ['stuck_joint', 'slippery_floor_easy', 'goal_shift_easy']
        self.damage_start_step = 500_000
        self.damage_steps = 1_000_000
        self.do_damage = True
        self.env = MockEnvironment()
        self.active_damages = set()
        
    def get_accumulative_damage_types(self, t):
        """Get all damage types that should be active at timestep t (accumulative)."""
        if not self.do_damage or t < self.damage_start_step:
            return []
        
        active_damages = []
        steps_since_damage_start = t - self.damage_start_step
        total_damage_duration = len(self.damage_type) * self.damage_steps
        
        # Check if we're still within the total damage sequence
        if steps_since_damage_start >= total_damage_duration:
            # All damages should remain active even after the sequence ends (true accumulation)
            return self.damage_type[:]
        
        # For each damage type, check if it should be active
        for i, damage_type in enumerate(self.damage_type):
            damage_start = i * self.damage_steps
            # Once a damage starts, it stays active (accumulative)
            if steps_since_damage_start >= damage_start:
                active_damages.append(damage_type)
        
        return active_damages
    
    def get_newly_introduced_damage(self, t):
        """Get the damage type that was just introduced at this timestep."""
        if not self.do_damage or t < self.damage_start_step:
            return None
        
        steps_since_damage_start = t - self.damage_start_step
        
        # Check if we're exactly at the start of a new damage cycle
        for i, damage_type in enumerate(self.damage_type):
            damage_start = i * self.damage_steps
            if steps_since_damage_start == damage_start:
                return damage_type
        
        return None
        
    def _initialize_damage(self, damage_type, t):
        """Initialize a specific damage type."""
        print(f"Step {t}: Initializing damage: {damage_type}")
        if damage_type == "slippery_floor":
            self.env.change_friction(-1.8, -1.8)
        elif damage_type == "slippery_floor_easy":
            self.env.change_friction(-1.7, -1.7)
        elif damage_type == "goal_shift":
            self.env.set_goal_offset(0, 3.0)
        elif damage_type == "goal_shift_easy":
            self.env.set_goal_offset(0, 2.4)
        # Action-based damages (broken_leg, stuck_joint) don't need environment initialization
        
    def simulate_training_step(self, t):
        """Simulate one training step's damage handling"""
        print(f"\n=== Step {t:,} ===")
        
        if self.do_damage:
            # Get all currently active damage types (accumulative)
            current_active_damages = self.get_accumulative_damage_types(t)
            newly_introduced_damage = self.get_newly_introduced_damage(t)
            
            print(f"Current active damages: {current_active_damages}")
            print(f"Newly introduced damage: {newly_introduced_damage}")
            
            # Initialize any newly introduced damage
            if newly_introduced_damage is not None:
                self._initialize_damage(newly_introduced_damage, t)
                self.active_damages.add(newly_introduced_damage)
            
            # Update the set of active damages
            self.active_damages.update(current_active_damages)
            
            print(f"All tracked active damages: {self.active_damages}")

def test_simulation():
    runner = MockRunner()
    
    # Test key steps where damages should be introduced
    test_steps = [
        500_000,    # First damage
        1_500_000,  # Second damage
        2_500_000,  # Third damage
        3_500_000,  # All should still be active
    ]
    
    for step in test_steps:
        runner.simulate_training_step(step)

if __name__ == "__main__":
    test_simulation()
