"""
Core functionality for Genie 2.0
"""


def hello():
    """
    Simple hello function.
    
    Returns:
        str: A greeting message
    """
    return "Hello from Genie 2.0!"


class Genie:
    """
    Main Genie class for the package.
    """
    
    def __init__(self, name="Genie"):
        """
        Initialize the Genie instance.
        
        Args:
            name (str): The name of the Genie instance
        """
        self.name = name
    
    def greet(self):
        """
        Greet the user.
        
        Returns:
            str: A personalized greeting
        """
        return f"Hello, I am {self.name}!"
