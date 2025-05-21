# BiLVL Project Code Index

## Core Components

### Models (`src/models/`)
- **robot.py**
  - `Robot4WSD`: Main robot class for 4-wheel steering and drive
    - `__init__(config)`: Initialize robot with configuration
    - `update(action)`: Update robot state based on control inputs
    - `kinematics(state, action)`: Compute robot kinematics (simplified model)
    - `dynamics(state, action)`: Compute robot dynamics based on physics model
    - `predict(state, action, dt)`: Predict next state for MPC
    - `_compute_slip_angles(state, delta_front, delta_rear)`: Calculate slip angles for each wheel
    - `_compute_wheel_velocities(state, delta_front, delta_rear)`: Calculate wheel velocities
    - `_compute_normal_forces()`: Calculate normal forces on each wheel
    - `_compute_wheel_positions()`: Calculate wheel positions relative to CG

- **motor.py**
  - `ElectricMotor`: Electric motor model with tire dynamics
    - `__init__(config)`: Initialize motor parameters from config
    - `update(voltage, wheel_velocity, slip_angle, normal_force, dt)`: Update motor state and calculate tire forces
    - `calc_tire_forces(slip_ratio, slip_angle, normal_force)`: Calculate tire forces using Pacejka magic formula

### Safety (`src/safety/`)
Safety is enforced through Control Barrier Functions (CBFs) located in `src/safety/barrier.py`. These include:
- `ControlBarrierFunction`: Base abstract class for all barrier functions
  - `__init__(config)`: Initialize CBF parameters from config
  - `get_adaptive_alpha()`: Get adaptive CBF parameter based on robot state
  - `get_uncertainty_margin()`: Calculate safety margin for uncertainty
  - `compute_robust_barrier()`: Compute barrier value with uncertainty
  - `h(human_state)`: Abstract method to evaluate the CBF h(x)
  - `h_dot(human_state)`: Abstract method to evaluate time derivative
  - `verify_safety(human_state)`: Check if CBF condition is satisfied

- `DistanceBarrier`: Distance-based safety (minimum safe distance: `rho_0 = 1.0m` front, `rho_1 = 0.5m` sides)
  - `_get_dynamic_safety_distance()`: Compute velocity-dependent safety distance
  - `h(human_state)`: Evaluate distance-based barrier function
  - `h_dot(human_state)`: Evaluate time derivative of distance CBF

- `YieldingBarrier`: Yielding behavior (base speed `2.0 m/s`, reduction starts at `2.0m`, full stop at `1.0m`)
  - `_smooth_activation()`: Compute smooth activation function using sigmoid
  - `_compute_max_speed()`: Calculate maximum allowed speed based on distance
  - `h(human_state)`: Evaluate yielding behavior barrier function
  - `h_dot(human_state)`: Evaluate time derivative of yielding CBF

- `SpeedBarrier`: Speed limits (base: `2.0 m/s`, slope: `-0.5 m/s per meter`)
  - `_get_dynamic_speed_limit()`: Compute dynamic speed limit based on state
  - `h(human_state)`: Evaluate speed limit barrier function
  - `h_dot(human_state)`: Evaluate time derivative of speed CBF

- `AccelBarrier`: Acceleration bounds (base: `1.0 m/s²`, slope: `-0.2 m/s² per meter`)
  - `_compute_max_accel()`: Calculate maximum allowed acceleration
  - `h(human_state)`: Evaluate acceleration bound barrier function
  - `h_dot(human_state)`: Evaluate time derivative of acceleration CBF

Implemented features:
- Adaptive CBF parameters based on robot state
- Smooth activation functions using sigmoid
- Velocity-dependent safety margins
- Uncertainty-aware bounds

Planned enhancements:
- Human prediction for proactive safety
- Recovery behaviors for constraint violations
- Online uncertainty estimation

### Control (`src/controllers/`)
- **mpc.py** (Primary controller - needs refactoring to match paper)
  - `MPCParams`: Data class for MPC configuration parameters
  - `BiLevelMPC`: Current implementation of Bi-Level MPC
    - `__init__(config)`: Initialize MPC controller with configuration
    - `action(state, desired_state)`: Compute control action using MPC
    - `solve_mpc(state, desired_state)`: Solve MPC optimization problem
    - `upper_level_cost(state, desired_state, control, dt, horizon, Q1, Q2, R, human_states)`: Calculate cost function
    - `calculate_gradient(state, desired_state, u, dt, Hp, Q1, Q2, R, human_states)`: Calculate gradient for optimization
    - `enforce_safety_constraints(state, u, human_states, dt, rho0, rho1, theta0, vmax)`: Enforce safety constraints
    
    **Required Refactoring:**
    - Implement true bi-level optimization structure as described in paper
    - Separate upper-level (performance) and lower-level (safety) optimization
    - Replace gradient-based approach with proper bi-level optimization solver
    - Ensure CBF constraints are correctly enforced at lower level

- **fast_mpc.py** (Alternative implementation - for comparison only)
  - `FastMPC`: Simplified MPC implementation for faster computation
    - Similar interface to BiLevelMPC but with optimized computation
    - Does not implement the paper's bi-level structure
    - Used for performance comparison only

- **simplified_mpc.py** (Alternative implementation - for comparison only)
  - `SimplifiedMPC`: Simplified MPC implementation without bi-level structure
    - Uses CasADi for optimization but lacks CBF constraints
    - Does not implement the paper's bi-level structure
    - Used for comparison with bi-level approach

- **pid.py**
  - `PIDParams`: Data class for PID parameters
  - `PID`: Proportional-Integral-Derivative controller
    - `__init__(config)`: Initialize PID controller with parameters
    - `action(error)`: Compute control action based on error
    - `reset()`: Reset controller state

Current implementation status:
- Basic bi-level optimization structure attempted in mpc.py
- Gradient-based optimization approach implemented
- Safety constraint enforcement partially implemented
- Multiple controller implementations exist but need consolidation

Required development:
- Refactor mpc.py to implement true bi-level optimization structure
- Replace current approach with proper bi-level optimization solver
- Consolidate useful features from all MPC implementations
- Ensure CBF time derivatives are correctly incorporated in constraints

### Simulation (`src/simulation/`)
- **simulator.py**
  - `Simulation`: Main simulation environment
    - `__init__(config)`: Initialize simulation with configuration
    - `step()`: Perform one simulation step
    - `is_running()`: Check if the scenario is still running

- **scenarios.py**
  - `BaseScenario`: Base scenario class
    - `__init__(config)`: Initialize scenario with configuration
    - `get_desired_state(time)`: Get desired state at given time
    - `get_initial_state()`: Get initial state for scenario
    - `get_human_positions()`: Get positions of humans in scenario

Implemented scenarios:
- Circle tracking
- Goal-directed navigation with static obstacles

Planned enhancements:
- Dynamic human motion models
- Social navigation metrics
- More complex scenarios with multiple humans
- Dynamic obstacles

### Visualization (`src/utils/`)
- **visualizer.py**
  - Basic visualization functions for robot state and performance metrics

- **mpc_visualization.py**
  - Visualization tools for MPC prediction and optimization

Implemented features:
- Basic robot state visualization
- Safety bounds visualization
- Performance metrics plotting

Planned enhancements:
- CBF visualization tools
- Safety boundary animation
- Real-time visualization of optimization process

## Current Development Status
The project has made significant progress in implementing the core components, but requires focused development on the bi-level optimization framework:

1. Robot Model:
   - Both kinematics and dynamics models are implemented
   - Electric motor model with Pacejka tire formula is complete
   - Basic uncertainty handling is in place

2. Safety Framework:
   - All four CBFs (distance, yielding, speed, acceleration) are implemented
   - Adaptive parameters and smooth activation functions are working
   - Basic uncertainty margins are incorporated

3. Controller Implementation:
   - Current BiLevelMPC implementation uses gradient-based optimization
   - **Does not fully match the paper's bi-level structure**
   - Multiple controller implementations exist (mpc.py, fast_mpc.py, simplified_mpc.py)
   - Need to consolidate into a single implementation matching the paper

4. Simulation Environment:
   - Basic simulation framework is operational
   - Circle tracking and goal-directed scenarios are implemented
   - Simple human obstacle representation is in place

## Next Development Priorities
1. **Implement true bi-level optimization structure as described in the paper**
   - Refactor mpc.py to match the paper's hierarchical structure
   - Replace gradient-based approach with proper bi-level optimization
   - Consolidate the three MPC implementations into a single coherent implementation

2. Enhance uncertainty handling in robot model and CBF constraints
   - Implement robust uncertainty estimation
   - Add parameter variation handling
   - Integrate uncertainty handling in MPC prediction

3. Develop comprehensive testing framework for validation
   - Create test scenarios matching the paper's examples
   - Implement validation for safety rules
   - Compare with traditional penalty-based approaches

## Reference Materials
The MATLAB files in the docs folder are reference materials only and not part of the Python implementation. They should be used as guidance for understanding the paper's approach but not directly incorporated into the codebase.
