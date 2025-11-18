#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the car evolution application core functions.
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Imports from src are handled by conftest.py
from app import (
    generate_random_vertices,
    get_random_car_gene,
    create_car,
    car_from_gene,
    destroy_car
)


class TestGenerateRandomVertices:
    """Test suite for generate_random_vertices function."""

    def test_returns_list(self):
        """Test that the function returns a list."""
        vertices = generate_random_vertices()
        assert isinstance(vertices, list)

    def test_vertices_are_tuples(self):
        """Test that each vertex is a tuple of two numbers."""
        vertices = generate_random_vertices()
        for vertex in vertices:
            assert isinstance(vertex, tuple)
            assert len(vertex) == 2
            assert isinstance(vertex[0], (int, float, np.floating))
            assert isinstance(vertex[1], (int, float, np.floating))

    def test_minimum_vertices(self):
        """Test that we get at least 3 vertices (minimum for a convex hull)."""
        vertices = generate_random_vertices()
        assert len(vertices) >= 3

    def test_vertices_within_expected_range(self):
        """Test that vertices are within the expected range [0, 3.0]."""
        vertices = generate_random_vertices()
        for x, y in vertices:
            assert 0 <= x <= 3.0
            assert 0 <= y <= 3.0


class TestGetRandomCarGene:
    """Test suite for get_random_car_gene function."""

    def test_returns_dict(self):
        """Test that the function returns a dictionary."""
        gene = get_random_car_gene()
        assert isinstance(gene, dict)

    def test_has_required_keys(self):
        """Test that the gene dict has all required keys."""
        gene = get_random_car_gene()
        required_keys = [
            'wheel_friction',
            'wheel_radius_1',
            'wheel_radius_2',
            'density',
            'wheel_torques',
            'wheel_drives',
            'motor_speeds',
            'hz',
            'zeta',
            'vertices'
        ]
        for key in required_keys:
            assert key in gene

    def test_wheel_friction_range(self):
        """Test that wheel_friction is in valid range [0, 1]."""
        gene = get_random_car_gene()
        assert 0 <= gene['wheel_friction'] <= 1

    def test_wheel_radii_range(self):
        """Test that wheel radii are in valid range [0, 1]."""
        gene = get_random_car_gene()
        assert 0 <= gene['wheel_radius_1'] <= 1
        assert 0 <= gene['wheel_radius_2'] <= 1

    def test_density_range(self):
        """Test that density is in valid range [0, 5]."""
        gene = get_random_car_gene()
        assert 0 <= gene['density'] <= 5.0

    def test_wheel_torques_length(self):
        """Test that wheel_torques has correct length."""
        gene = get_random_car_gene()
        assert len(gene['wheel_torques']) == 2
        for torque in gene['wheel_torques']:
            assert 0 <= torque <= 20

    def test_wheel_drives_length(self):
        """Test that wheel_drives has correct length."""
        gene = get_random_car_gene()
        assert len(gene['wheel_drives']) == 2
        assert all(isinstance(x, bool) for x in gene['wheel_drives'])

    def test_motor_speeds_length(self):
        """Test that motor_speeds has correct length."""
        gene = get_random_car_gene()
        assert len(gene['motor_speeds']) == 2
        for speed in gene['motor_speeds']:
            assert -20.0 <= speed <= 0.0

    def test_hz_range(self):
        """Test that hz is in valid range."""
        gene = get_random_car_gene()
        assert 0 <= gene['hz'] <= 15.0

    def test_zeta_range(self):
        """Test that zeta is in valid range [0, 1]."""
        gene = get_random_car_gene()
        assert 0 <= gene['zeta'] <= 1


class TestCreateCar:
    """Test suite for create_car function."""

    @pytest.fixture
    def mock_world(self):
        """Create a mock Box2D world."""
        world = Mock()

        # Mock for dynamic body creation
        mock_body = Mock()
        mock_body.position = Mock(x=0, y=0)
        world.CreateDynamicBody.return_value = mock_body

        # Mock for wheel joint creation
        mock_joint = Mock()
        world.CreateWheelJoint.return_value = mock_joint

        return world

    def test_creates_car_with_default_params(self, mock_world):
        """Test creating a car with default parameters."""
        chassis, wheels, springs = create_car(
            mock_world,
            offset=(0, 0),
            wheel_radius_1=0.5,
            wheel_radius_2=0.5,
            wheel_separation=2.0
        )

        assert chassis is not None
        assert len(wheels) == 2
        assert len(springs) == 2
        assert mock_world.CreateDynamicBody.call_count == 3  # 1 chassis + 2 wheels
        assert mock_world.CreateWheelJoint.call_count == 2  # 2 springs

    def test_creates_car_with_custom_vertices(self, mock_world):
        """Test creating a car with custom chassis vertices."""
        custom_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        chassis, wheels, springs = create_car(
            mock_world,
            offset=(0, 0),
            wheel_radius_1=0.5,
            wheel_radius_2=0.5,
            wheel_separation=2.0,
            chassis_vertices=custom_vertices
        )

        assert chassis is not None
        assert len(wheels) == 2
        assert len(springs) == 2


class TestCarFromGene:
    """Test suite for car_from_gene function."""

    @pytest.fixture
    def mock_world(self):
        """Create a mock Box2D world."""
        world = Mock()
        mock_body = Mock()
        mock_body.position = Mock(x=0, y=0)
        world.CreateDynamicBody.return_value = mock_body
        mock_joint = Mock()
        world.CreateWheelJoint.return_value = mock_joint
        return world

    def test_creates_car_from_gene(self, mock_world):
        """Test creating a car from a gene dictionary."""
        gene = get_random_car_gene()
        chassis, wheels, springs = car_from_gene(mock_world, gene)

        assert chassis is not None
        assert len(wheels) == 2
        assert len(springs) == 2


class TestDestroyCar:
    """Test suite for destroy_car function."""

    def test_destroys_car_bodies(self):
        """Test that destroy_car calls DestroyBody for all car components."""
        mock_world = Mock()
        mock_car = {
            'car': Mock(),
            'wheels': [Mock(), Mock()]
        }

        destroy_car(mock_world, mock_car)

        assert mock_world.DestroyBody.call_count == 3
        mock_world.DestroyBody.assert_any_call(mock_car['car'])
        mock_world.DestroyBody.assert_any_call(mock_car['wheels'][0])
        mock_world.DestroyBody.assert_any_call(mock_car['wheels'][1])


class TestIntegration:
    """Integration tests for car generation workflow."""

    @pytest.fixture
    def mock_world(self):
        """Create a mock Box2D world."""
        world = Mock()
        mock_body = Mock()
        mock_body.position = Mock(x=0, y=0)
        world.CreateDynamicBody.return_value = mock_body
        mock_joint = Mock()
        world.CreateWheelJoint.return_value = mock_joint
        return world

    def test_full_car_lifecycle(self, mock_world):
        """Test complete car creation and destruction workflow."""
        # Generate a gene
        gene = get_random_car_gene()

        # Create car from gene
        chassis, wheels, springs = car_from_gene(mock_world, gene)

        # Create car dict
        car = {
            'car': chassis,
            'wheels': wheels,
            'springs': springs,
            'gene': gene
        }

        # Destroy car
        destroy_car(mock_world, car)

        # Verify destruction was called
        assert mock_world.DestroyBody.call_count == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
