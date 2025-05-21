# 2D Robot Simulation Environment

A modular and expandable framework for simulating robotic vehicles in a 2D environment. The framework supports various controllers and robot models, making it ideal for testing and development in robotics research and education.

## Description

This framework provides a 2D simulation environment with built-in support for PID and MPC controllers, as well as kinematic and dynamic solvers. It is designed to be modular and easily extendable, allowing users to:

1. Add new robot models with different kinematic and dynamic properties.
2. Implement and test additional controller types.
3. Create and evaluate custom simulation scenarios.

### Key Features

- **Modular Design**:
  - Easily integrate new robot models.
  - Add custom controllers.
  - Define and test new scenarios.

- **Controller Support**:
  - PID controller for simple and effective control.
  - MPC controller for advanced trajectory optimization.

- **Simulation Capabilities**:
  - Kinematic and dynamic solvers for accurate robot behavior.
  - Configurable simulation parameters.
  - Real-time visualization of robot motion and control outputs.

- **Advanced Robot Control**:
  - 4-wheel steering and drive (4WSD) support
  - Motor dynamics with voltage-based control
  - Tire model with Pacejka magic formula
  - Uncertainty-aware robust control

- **Flexible Architecture**:
  - Configurable simulation scenarios
  - Modular controller design (MPC/PID)
  - Real-time visualization
  - Extensive parameter tuning options

## Prerequisites

Before installing, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tools (optional but recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SDSNT8810/Robot_Simulator.git
cd Robot_Simulator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -e .
```

## Usage

1. Configure simulation parameters in `config/config.yaml`

2. Run a basic simulation:
```bash
python main.py
```

3. Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

4. Override specific parameters:
```bash
python main.py --param scenario.name to_goal --param controller.type PID
```

### Configuration Options

Key configuration sections in `config/config.yaml`:
- `robot`: Physical parameters and motor specifications
- `controller`: PID/MPC parameters and constraints
- `scenario`: Test scenarios and environment setup

## Project Structure

```
Robot_Simulator/
├── config/              # Configuration files
├── docs/                # Documentation
├── logs/                # Log files
├── runs/                # Simulation run outputs
├── src/                 # Source code
│   ├── controllers/     # MPC and PID implementations
│   ├── models/          # Robot and motor models
│   ├── safety/          # Control Barrier Functions
│   ├── simulation/      # Simulation environment
│   └── utils/           # Helper functions and visualization
├── tools/               # Additional tools and scripts
├── LICENSE              # License file
├── main.py              # Entry point for the simulation
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── setup.py             # Installation script
└── simulation_result.png # Example simulation result
```

## Development Roadmap

Current focus areas:
1. Modular robot model integration
2. Computational efficiency improvements
3. Testing framework expansion

See `ToDo.md` for detailed development plans.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [License](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{arab2025bilevel,
  title={Bi-Level Performance-Safety Consideration in Nonlinear Model Predictive Control},
  author={Arab, Aliasghar and Jalaeian-Farimani, Mohsen and Khalili-Amirabadi, Roya, and Nikkhouy, Davoud and Rastegarmoghaddam, Mahshad},
  journal={Proceedings of Machine Learning Research},
  year={2025}
}
```

## Sponsorship
This project is supported by professor Mohsen Jalaeian Farimani at the [Politecnico di Milano](https://www.polimi.it/en/) and professor Aliasghar Arab at the [NYU Tandon School of Engineering](https://engineering.nyu.edu/), and Dr. Roya Khalili Amirabadi at the [Ferdowsi University of Mashhad](https://en.um.ac.ir/). The project is part of the research activities in the field of robotics and control systems.


## Authors
- **Davoud Nikkhouy** - Politecnico di Milano
- **Mohsen Jalaeian-Farimani** - Politecnico di Milano
- **Mahshad Rastegarmoghaddam** - Politecnico di Milano