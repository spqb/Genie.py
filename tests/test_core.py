"""
Tests for core functionality.
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from genie.core import hello, Genie


def test_hello():
    """Test the hello function."""
    result = hello()
    assert result == "Hello from Genie 2.0!"
    assert isinstance(result, str)


def test_genie_init():
    """Test Genie initialization."""
    genie = Genie()
    assert genie.name == "Genie"
    
    genie_custom = Genie(name="CustomGenie")
    assert genie_custom.name == "CustomGenie"


def test_genie_greet():
    """Test Genie greet method."""
    genie = Genie(name="TestGenie")
    result = genie.greet()
    assert result == "Hello, I am TestGenie!"
    assert isinstance(result, str)


def test_genie_default_greet():
    """Test Genie greet with default name."""
    genie = Genie()
    result = genie.greet()
    assert result == "Hello, I am Genie!"
