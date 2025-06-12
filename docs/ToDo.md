# Optimization Framework Implementation Plan

## Phase 1: Core Components Implementation
- [ ] Robot Model (src/models/robot.py)
  - [x] Basic state representation
  - [x] Kinematics model implementation
  - [x] Dynamics model implementation
  - [x] Tire model with Pacejka magic formula
    - [x] Longitudinal slip ratio calculation
    - [x] Lateral slip angle calculation
    - [x] Friction circle model
  - [x] Motor dynamics (src/models/motor.py)
    - [x] Electrical dynamics (voltage, current, torque)
    - [x] Gear ratio and wheel coupling
  - [ ] Uncertainty bounds
    - [x] Basic uncertainty model for slip effects
    - [ ] Enhance parameter variation handling
    - [ ] Implement robust uncertainty estimation

- [x] Safety Framework (src/safety/barrier.py)
  - [x] Distance-based CBF
  - [x] Yielding behavior CBF
  - [x] Speed limit CBF
  - [x] Acceleration bound CBF
  - [x] CBF condition verification
  - [x] Basic adaptive safety margins
  - [ ] Enhance dynamic uncertainty estimation
  - [ ] Add human prediction models for proactive safety

- [ ] Controller Implementation (src/controllers/mpc.py)
  - [x] Basic BiLevelMPC class structure
  - [x] State prediction model
  - [x] Cost function structure
  - [ ] **True Bi-Level Optimization Structure**
    - [ ] Refactor to match paper's hierarchical structure
    - [ ] Separate upper-level (performance) and lower-level (safety) optimization
    - [ ] Implement proper bi-level optimization solver
    - [ ] Ensure CBF constraints are enforced at lower level
  - [ ] Optimization solver refinement
    - [x] Basic gradient-based solver
    - [ ] Replace with proper QP/NLP solver (e.g., CVXPY or OSQP)
    - [ ] Implement constraint linearization for CBFs
  - [ ] Robust MPC implementation
    - [ ] Add uncertainty handling in prediction
    - [ ] Implement tube-based robust MPC approach
  - [ ] Consolidate MPC implementations
    - [ ] Merge useful features from fast_mpc.py and simplified_mpc.py
    - [ ] Ensure single, coherent implementation matching paper

## Phase 2: Simulation Environment
- [ ] Scenario Implementation (src/simulation/scenarios.py)
  - [x] Basic scenario framework
  - [x] Circle tracking scenario
  - [x] Goal-directed navigation scenario
  - [ ] Enhanced human obstacle interaction
    - [x] Basic human position representation
    - [ ] Implement dynamic human motion models
    - [ ] Add social navigation metrics
  - [ ] Add more complex scenarios (multiple humans, dynamic obstacles)

- [ ] Visualization (src/utils/visualizer.py)
  - [x] Basic robot state visualization
  - [x] Safety bounds visualization
  - [x] Performance metrics plotting
  - [ ] Add CBF visualization tools
  - [ ] Implement safety boundary animation

## Phase 3: Testing and Validation
- [ ] Unit Tests (src/tests/)
  - [ ] Robot model tests
    - [ ] Test tire model with different friction conditions
    - [ ] Test motor dynamics with varying loads
    - [ ] Validate uncertainty bounds with perturbation analysis
  - [ ] CBF verification tests
    - [ ] Test individual barriers with edge cases
    - [ ] Test combined safety constraints for conflicts
    - [ ] Verify robustness to uncertainty
  - [ ] Controller performance tests
    - [ ] Test MPC optimization convergence
    - [ ] Validate safety guarantees under disturbances
    - [ ] Benchmark computational performance
  - [ ] Scenario execution tests
    - [ ] Test trajectory tracking with varying parameters
    - [ ] Test human interaction with different behaviors
    - [ ] Evaluate safety-performance tradeoffs

## Phase 4: Performance Optimization
- [ ] Computational Efficiency
  - [ ] Optimize MPC formulation for real-time execution
  - [ ] Implement sparse solver exploitation
  - [ ] Add parallel computation for CBF evaluations
  - [ ] Profile and optimize computational bottlenecks
  - [ ] Implement adaptive prediction horizon

- [ ] Safety Framework Enhancements
  - [x] Add smooth activation functions for barriers
  - [x] Implement velocity-dependent adaptive CBF parameters
  - [ ] Add recovery behaviors for constraint violations
  - [ ] Improve uncertainty handling with online estimation

## Phase 5: Documentation and Examples
- [x] API Documentation
  - [x] Document robot model interface
  - [x] Document safety framework
  - [x] Document controller interface
  - [ ] Add usage examples with code snippets
  - [ ] Document mathematical foundations in code comments

- [ ] Usage Guide
  - [ ] Installation instructions and dependencies
  - [ ] Configuration guide for different scenarios
  - [ ] Parameter tuning guidelines with practical examples
  - [ ] Example scenarios with expected outputs
  - [ ] Troubleshooting common issues

## Current Focus
1. **Implement True Bi-Level Optimization Structure**
   - Refactor mpc.py to match the paper's hierarchical structure with:
     - Upper level: Minimize tracking error and control effort
     - Lower level: Maximize safety through CBF constraints
   - Replace current gradient-based approach with proper bi-level optimization
   - Ensure CBF time derivatives are correctly incorporated in constraints
   - Consolidate the three MPC implementations (mpc.py, fast_mpc.py, simplified_mpc.py) into a single coherent implementation

2. Complete Robust Uncertainty Handling
   - Enhance uncertainty bounds in robot model
   - Implement parameter variation handling
   - Add robust uncertainty estimation for CBF constraints
   - Integrate uncertainty handling in MPC prediction

3. Develop Testing Framework
   - Create simulation scenarios matching the paper's validation examples
   - Implement specific test cases for safety rules in Table 2
   - Add visualization for safety constraints and barrier functions 
   - Create performance benchmarks against traditional penalty-based approaches

## Notes
- The robot model (Robot4WSD) has both kinematics and dynamics implementations
- The motor model (ElectricMotor) is fully implemented with Pacejka tire model
- Safety barriers (CBFs) are implemented with adaptive parameters
- Current BiLevelMPC implementation uses gradient-based optimization but does not fully match the paper's bi-level structure
- FastMPC and SimplifiedMPC are alternative implementations for comparison, not matching the paper's requirements
- MATLAB files in the docs folder are reference materials only, not part of the implementation
- Focus should be on implementing the true bi-level optimization structure as described in the paper
