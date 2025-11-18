#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the state module.
"""
import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from state import State


class TestState:
    """Test suite for State dataclass."""

    def test_state_default_values(self):
        """Test that State has correct default values."""
        state = State()

        assert state.fps == 0
        assert state.num_bodies == 0
        assert state.num_joints == 0
        assert state.num_proxies == 0
        assert state.num_contacts == 0
        assert state.cars is None
        assert state.car_pos == 0.0
        assert state.draw_aabb is False
        assert state.draw_obb is False
        assert state.draw_joints is False
        assert state.draw_pairs is False
        assert state.draw_coms is False
        assert state.paused is False
        assert state.reset is False
        assert state.generation == 0

    def test_state_custom_values(self):
        """Test that State can be initialized with custom values."""
        state = State(
            fps=60,
            num_bodies=10,
            paused=True,
            generation=5
        )

        assert state.fps == 60
        assert state.num_bodies == 10
        assert state.paused is True
        assert state.generation == 5

    def test_state_attribute_modification(self):
        """Test that State attributes can be modified."""
        state = State()

        state.fps = 120
        state.num_bodies = 25
        state.paused = True
        state.generation = 3

        assert state.fps == 120
        assert state.num_bodies == 25
        assert state.paused is True
        assert state.generation == 3

    def test_state_boolean_flags(self):
        """Test that all boolean flags can be toggled."""
        state = State()

        # Test draw flags
        state.draw_aabb = True
        state.draw_obb = True
        state.draw_joints = True
        state.draw_pairs = True
        state.draw_coms = True

        assert state.draw_aabb is True
        assert state.draw_obb is True
        assert state.draw_joints is True
        assert state.draw_pairs is True
        assert state.draw_coms is True

        # Test control flags
        state.paused = True
        state.reset = True

        assert state.paused is True
        assert state.reset is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
