# BiLVL Project Code Index

## Core Components

### Models (`src/models/`)
- **robot.py**
  - `Robot4WSD`: Main robot class for 4-wheel steering and drive
    - `__init__(config)`: Initialize robot with configuration
    - `nominal_dynamics(state, input_)`: Compute nominal dynamics
    - `uncertainty(state, input_)`: Compute uncertainty bounds
    - `_compute_tire_forces(slip_angles, slip_ratios, normal_forces)`: Calculate tire forces
    - `_compute_slip_angles(state)`: Calculate slip angles for each wheel
    - `_compute_wheel_positions()`: Calculate wheel positions relative to CG
    - `step(dt)`: Step simulation forward by dt

- **state.py**
  - `RobotState`: Robot state representation
    - State vector z = [q^T v^T I^T]
    - q: pose [x, y, θ]
    - v: velocity [vx, vy, ω]
    - I: motor currents [I_FL, I_FR, I_RL, I_RR]

- **input.py**
  - `RobotInput`: Robot input representation
    - Input vector u = [δ^T V^T]
    - δ: steering angles [δ_front, δ_rear]
    - V: motor voltages [V_FL, V_FR, V_RL, V_RR]

- **motor.py**
  - `ElectricMotor`: Electric motor model
    - `compute_torque(voltage, speed)`: Get motor torque
    - `update_current(voltage, speed, dt)`: Update motor current

### Safety (`src/safety/`)
Safety is enforced through Control Barrier Functions (CBFs) located in `src/safety/barrier.py`. These include:
- Distance-based safety (minimum safe distance: `rho_0 = 1.0m` front, `rho_1 = 0.5m` sides)
- Yielding behavior (base speed `2.0 m/s`, reduction starts at `2.0m`, full stop at `1.0m`)
- Speed limits (base: `2.0 m/s`, slope: `-0.5 m/s per meter`)
- Acceleration bounds (base: `1.0 m/s²`, slope: `-0.2 m/s² per meter`)

Planned enhancements:
- Dynamic CBF parameters (in progress)
  - Velocity-dependent safety margins
  - Adaptive barrier function parameters
  - Uncertainty-aware bounds
- Smooth activation functions
- Recovery behaviors

### Control (`src/controllers/`)
- **mpc.py**
  - `ModelPredictiveControl`: QP-based MPC implementation
    - `predict_trajectory(state, input_sequence)`: Forward prediction
    - `compute_cost(predicted_states, predicted_inputs)`: Cost function
    - `solve_optimization(state, reference)`: Solve QP-based MPC problem
    - `update_qp_matrices(current_state, reference_trajectory)`: Update QP matrices for optimization

Planned optimizations:
- QP solver warm-starting
- Efficient constraint handling
- State estimation
- Matrix computation optimization

### Simulation (`src/simulation/`)
- **simulator.py**
  - `Simulator`: Main simulation environment
    - Handles time stepping and physics
    - Manages scenario execution
    - Visualization integration

- **scenarios.py**
  - `TestScenario`: Base scenario class
    - Handles safety monitoring and metrics
    - Loads parameters from config
    - CBF-based safety enforcement
  - `SafeTrajectoryTracking`: Main test scenario
    - Tests bi-level control framework
    - Evaluates safety-performance tradeoff
    - Configurable path generation and control gains

### Configuration (`config/config.yaml`)
Centralized configuration with clear parameter hierarchy:
- **Robot Parameters**: Physical and mechanical properties
- **Safety Parameters**: CBF thresholds and constraints
- **Controller Parameters**: MPC and control gains
- **Visualization Parameters**: Display settings
- **Simulation Parameters**: Time steps and initial conditions
- **Scenario Parameters**:
  - Common parameters across scenarios
  - Safety monitoring settings
  - Trajectory tracking parameters
  - Path generation configuration
  - Control gains and thresholds

### Tests (`src/tests/`)
- **test_robot_model.py**: Unit tests for robot dynamics
- **test_safety.py**: Unit tests for CBF implementation
- **test_mpc.py**: Unit tests for QP-based MPC implementation

### Utils (`src/utils/`)
- **config.py**: Configuration management and parameter access

## Examples
- **scenario_runner.py**: Evaluates the bi-level control framework
  - Runs safe trajectory tracking tests
  - Reports performance and safety metrics
  - Real-time visualization of results

## Documentation
- **docs/Description.md**: Mathematical foundations
- **docs/INDEX.md**: This file - code structure
- **ToDo.md**: Project roadmap

## Current Development Focus
The project is currently focused on implementing dynamic safety parameters and improving computational efficiency:

1. Safety Enhancements:
   - Dynamic CBF parameters based on velocity and state
   - Adaptive safety margins with uncertainty consideration
   - Smooth activation functions for constraints

2. Performance Optimization:
   - QP solver warm-starting implementation
   - Constraint handling efficiency
   - State estimation with EKF
   - Matrix computation optimization

3. Testing Framework:
   - QP solver unit tests
   - Constraint handling stress tests
   - Safety visualization tools
   - Real-time performance benchmarks
