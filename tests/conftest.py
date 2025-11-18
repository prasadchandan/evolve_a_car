#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures.
This file is automatically loaded by pytest before any tests run.
"""
import sys
import os
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# Add src directory to path before any imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock GUI libraries to prevent initialization issues during test collection
# These must be mocked BEFORE the modules are imported

# Mock pyglet
sys.modules['pyglet'] = MagicMock()
sys.modules['pyglet.gl'] = MagicMock()
sys.modules['pyglet.window'] = MagicMock()
sys.modules['pyglet.window.key'] = MagicMock()
sys.modules['pyglet.window.mouse'] = MagicMock()
sys.modules['pyglet.graphics'] = MagicMock()
sys.modules['pyglet.text'] = MagicMock()

# Mock imgui
sys.modules['imgui'] = MagicMock()
sys.modules['imgui.core'] = MagicMock()
sys.modules['imgui.integrations'] = MagicMock()
sys.modules['imgui.integrations.pyglet'] = MagicMock()

# Mock OpenGL
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()
sys.modules['OpenGL.GLU'] = MagicMock()
sys.modules['OpenGL.GLUT'] = MagicMock()

# Create mock settings module to prevent optparse conflicts
# This must be done before any src modules are imported
class MockSettings:
    """Mock settings module to avoid optparse conflicts with pytest."""

    @dataclass
    class config:
        font_name: str = './resources/Gidole-Regular.ttf'
        mono_font_name: str = './resources/Hack-Regular.ttf'
        font_size: int = 16

    class fwSettings:
        backend = 'pyglet'
        screen_width = 800
        screen_height = 600
        hz = 60.0
        velocityIterations = 8
        positionIterations = 3
        enableWarmStarting = True
        enableContinuous = True
        enableSubStepping = False
        drawStats = False
        drawShapes = True
        drawJoints = True
        drawCoreShapes = False
        drawAABBs = False
        drawOBBs = False
        drawPairs = False
        drawContactPoints = False
        maxContactPoints = 100
        drawContactNormals = False
        drawFPS = True
        drawMenu = True
        drawCOMs = False
        pointSize = 2.5
        pause = False
        singleStep = False
        onlyInit = False

    checkboxes = (("Warm Starting", "enableWarmStarting"),)
    sliders = []

mock_settings = MockSettings()
sys.modules['settings'] = mock_settings

# Mock framework and other GUI-dependent modules
mock_framework = MagicMock()
mock_framework.Keys = MagicMock()
mock_framework.Keys.K_a = 'a'
mock_framework.Keys.K_s = 's'
mock_framework.Keys.K_d = 'd'
mock_framework.Keys.K_q = 'q'
mock_framework.Keys.K_e = 'e'
mock_framework.Keys.K_r = 'r'
sys.modules['framework'] = mock_framework

# Mock renderer
sys.modules['renderer'] = MagicMock()

# Mock UI
sys.modules['ui'] = MagicMock()
