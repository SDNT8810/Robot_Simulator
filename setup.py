from setuptools import setup, find_packages

setup(
    name="bilevel-safe-control",
    version="0.1.0",
    description="Bi-Level Performance-Safety Control Framework for Robotic Vehicles",
    author="Aliasghar Arab, Mohsen Jalaeian Farimani, Roya Khalili Amirabadi",
    author_email="aliasghar.arab@nyu.edu",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'scipy>=1.11.0',
        'osqp>=0.6.3',  # For QP-based MPC optimization
        'pyyaml>=6.0.1',  # For config file handling
        'pytest>=8.0.0',  # For testing
        'cvxopt>=1.3.0',  # For QP solving
        'control>=0.9.0',  # For control system analysis
        'casadi>=3.5.5'  # For nonlinear optimization
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Control',
        'Topic :: Scientific/Engineering :: Robotics'
    ]
)
