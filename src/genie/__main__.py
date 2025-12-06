"""
Main entry point for Genie when run as a module.
"""

from .core import hello


def main():
    """Main function."""
    print(hello())


if __name__ == "__main__":
    main()
