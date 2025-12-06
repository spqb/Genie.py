from setuptools import setup, find_packages
import os

# Read the version from __version__.py
version = {}
with open(os.path.join("src", "genie", "__version__.py")) as fp:
    exec(fp.read(), version)

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genie2",
    version=version["__version__"],
    author="Roberto Netti",
    description="Genie 2.0 - A Python Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robertonetti/Genie2.0.py",
    project_urls={
        "Bug Tracker": "https://github.com/robertonetti/Genie2.0.py/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "genie=genie.__main__:main",
        ],
    },
)
