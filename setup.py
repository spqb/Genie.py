from setuptools import setup, find_packages

setup(
    name="genie",
    version="2.0.0",
    description="Genie 2.0 Application",
    author="Roberto Netti",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "adabmDCA",
    ],
    entry_points={
        "console_scripts": [
            "genie=Genie.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
