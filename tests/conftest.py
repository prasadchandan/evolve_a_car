#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures.
This file is automatically loaded by pytest before any tests run.
"""
import sys
import os
from unittest.mock import Mock, MagicMock

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
