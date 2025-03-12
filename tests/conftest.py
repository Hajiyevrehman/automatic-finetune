import os
import sys
# Add the project root directory to Python's module search path
# Going up one level from tests directory to get to project root
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))