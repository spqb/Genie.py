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
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.0.0",
        "scikit-learn>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "genie=Genie.main:main",
            "genie-aa=Genie_aa.main:main",
            "reconstruct_chains=scripts.reconstruct_chains:main",
            "reconstruct_at_timesteps=scripts.reconstruct_at_timesteps:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
