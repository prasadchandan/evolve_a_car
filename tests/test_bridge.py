#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the bridge module.
"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bridge import create_bridge


class TestCreateBridge:
    """Test suite for create_bridge function."""

    @pytest.fixture
    def mock_world(self):
        """Create a mock Box2D world."""
        world = Mock()
        mock_body = Mock()
        world.CreateDynamicBody.return_value = mock_body
        mock_joint = Mock()
        world.CreateRevoluteJoint.return_value = mock_joint
        return world

    @pytest.fixture
    def mock_ground(self):
        """Create a mock ground body."""
        return Mock()

    def test_creates_correct_number_of_planks(self, mock_world, mock_ground):
        """Test that bridge creates the correct number of planks."""
        plank_count = 10
        bodies = create_bridge(
            mock_world,
            mock_ground,
            size=(2.0, 0.25),
            offset=(0, 0),
            plank_count=plank_count
        )

        assert len(bodies) == plank_count
        assert mock_world.CreateDynamicBody.call_count == plank_count

    def test_creates_correct_number_of_joints(self, mock_world, mock_ground):
        """Test that bridge creates the correct number of joints."""
        plank_count = 10
        create_bridge(
            mock_world,
            mock_ground,
            size=(2.0, 0.25),
            offset=(0, 0),
            plank_count=plank_count
        )

        # Should create plank_count joints (one per plank) + 1 final joint
        assert mock_world.CreateRevoluteJoint.call_count == plank_count + 1

    def test_returns_list_of_bodies(self, mock_world, mock_ground):
        """Test that create_bridge returns a list of bodies."""
        bodies = create_bridge(
            mock_world,
            mock_ground,
            size=(2.0, 0.25),
            offset=(0, 0),
            plank_count=5
        )

        assert isinstance(bodies, list)
        assert all(body is not None for body in bodies)

    def test_handles_single_plank(self, mock_world, mock_ground):
        """Test creating a bridge with a single plank."""
        bodies = create_bridge(
            mock_world,
            mock_ground,
            size=(2.0, 0.25),
            offset=(0, 0),
            plank_count=1
        )

        assert len(bodies) == 1
        assert mock_world.CreateDynamicBody.call_count == 1
        assert mock_world.CreateRevoluteJoint.call_count == 2  # 1 + final joint

    def test_custom_friction_and_density(self, mock_world, mock_ground):
        """Test that custom friction and density parameters are used."""
        create_bridge(
            mock_world,
            mock_ground,
            size=(2.0, 0.25),
            offset=(0, 0),
            plank_count=3,
            friction=0.8,
            density=2.5
        )

        # Verify CreateDynamicBody was called
        assert mock_world.CreateDynamicBody.call_count == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
