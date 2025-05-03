# Bi-Level Safe Control Framework

A hierarchical optimization framework for controlling robotic vehicles with explicit safety guarantees, particularly designed for active front and rear wheel steering vehicles operating in human-centric environments.

## Description

This framework introduces a novel bi-level optimization approach integrated with Nonlinear Model Predictive Control (NMPC) to address the critical trade-off between performance and safety. Instead of treating safety as a penalty term in the cost function, which can lead to suboptimal performance, the framework decomposes the control problem into two interdependent levels:

1. **Upper Level**: Uses MPC to minimize a cost function focused on trajectory tracking accuracy and actuation efficiency
2. **Lower Level**: Explicitly maximizes safety through Control Barrier Functions (CBFs)

### Key Features

- **Comprehensive Safety Framework**: 
  - Distance-based safety with dynamic thresholds
  - Yielding behavior in human proximity
  - Adaptive speed and acceleration limits
  - Real-time CBF constraint enforcement

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
- `safety`: CBF parameters and safety thresholds
- `controller`: MPC/PID parameters and constraints
- `scenario`: Test scenarios and environment setup

## Project Structure

```
Robot_Simulator/
├── src/
│   ├── models/          # Robot and motor models
│   ├── controllers/     # MPC and PID implementations
│   ├── safety/         # Control Barrier Functions
│   ├── simulation/     # Simulation environment
│   ├── visualization/  # Real-time plotting
│   └── utils/         # Helper functions
├── config/            # Configuration files
├── docs/             # Documentation
└── examples/         # Usage examples
```

## Development Roadmap

Current focus areas:
1. Dynamic safety parameter implementation
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
- **Mohsen Jalaeian-Farimani** - Politecnico di Milano
- **Davoud Nikkhouy** - Politecnico di Milano
- **Mahshad Rastegarmoghaddam** - Politecnico di Milano