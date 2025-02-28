"""
Smoke test to verify that the basic project setup is working.
"""
import os
import sys


def test_python_version():
    """Verify Python version is at least 3.10"""
    major, minor = sys.version_info[:2]
    assert (major, minor) >= (
        3,
        10,
    ), f"Python version must be at least 3.10, got {major}.{minor}"


def test_environment_setup():
    """Verify that we can import core dependencies"""
    try:
        import logging
    except ImportError as e:
        assert False, f"Failed to import core dependency: {e}"


def test_project_structure():
    """Verify essential project directories exist"""
    essential_dirs = ["data", "src", "configs", "tests"]
    for directory in essential_dirs:
        assert os.path.isdir(directory), f"Required directory '{directory}' not found"
