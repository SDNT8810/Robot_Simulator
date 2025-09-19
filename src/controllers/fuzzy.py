"""Fuzzy-style controller with goal-directed free-space selection.

Implements a MATLAB-style fuzzy angular sector evaluation. Each sector gets:
    Sec_Free_Val[j]  ~ average normalized LIDAR distance within Gaussian window
    Goal_Dir_Value[j] = w_goal(center_j) * Sec_Free_Val[j]

Where w_goal is a Gaussian centered at the goal direction (in robot frame).
We pick the sector index argmax(Goal_Dir_Value); if all weights are ~0 we
fallback to argmax(Sec_Free_Val).

Output format (same as PID):
        [delta_front, delta_rear, V_fl, V_fr, V_rl, V_rr]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

from src.models.fuzzyFunctions import FuzzyFunctions, FuzzyParams
from typing import Dict, Any


@dataclass
class FuzzyParamsController:
    constant_speed: float = 1.8           # Target forward speed (m/s)
    steer_gain: float = 0.4              # Positive: turn toward positive heading error
    lateral_gain: float = 0.0             # (Optional) lateral correction not used here
    max_steer: float = 0.5                # Steering limit (rad)
    smoothing_factor: float = 0.1         # Voltage smoothing factor
    front_rear_scale: float = -0.8        # Rear/front steering ratio (like PID config)
    deadband_heading: float = 0.01        # Deadband on heading error (rad)
    # Removed goal weighting: always choose direction of maximum Sec_Free_Val


class Fuzzy:
    """Reactive controller using fuzzy section free-space selection.

    Public API identical to PID: action(state, desired_state, safety_data) -> control vector.
    Internally ignores the provided desired_state position and synthesizes its own heading target
    based on LIDAR Sec_Free_Val (max section center direction).
    """

    def __init__(self, config: dict, ctrl_params: Optional[FuzzyParamsController] = None):
        self.config = config
        self.cparams = ctrl_params or FuzzyParamsController()
        # last decision cache for external inspection
        self.last_decision = {}

        # Fuzzy membership helper configuration (can be overridden via config['controller']['fuzzy'])
        overrides = config.get('controller', {}).get('fuzzy', {})
        fp = FuzzyParams(
            NL=overrides.get('NL', 360),
            Ns=overrides.get('Ns', 12),
            std_deg=overrides.get('std_deg', 20.0),
            G_dirc_deg=overrides.get('G_dirc_deg', -160.0),
            G_STD_deg=overrides.get('G_STD_deg', 70.0),
        )
        self.ff = FuzzyFunctions(fp)

        # Robot params
        self.wheelbase = config['robot']['wheelbase']
        self.track_width = config['robot']['track_width']
        self.max_steer_hw = config['controller']['mpc']['constraints']['steering']['max'][0]

        # Motor params
        motor_cfg = config['robot']['motor']
        self.voltage_speed_factor = motor_cfg.get('voltage_speed_factor', 0.1)
        self.motor_efficiency = motor_cfg.get('efficiency', 0.9)
        self.gear_ratio = motor_cfg.get('gear_ratio', 15.0)
        self.v_limit = motor_cfg.get('max_voltage', 12.0)

        pid_cfg = config['controller']['pid']
        min_out = pid_cfg.get('min_output', -1.0)
        max_out = pid_cfg.get('max_output', 1.0)
        self.u_min = np.min(min_out) if isinstance(min_out, (list, tuple)) else min_out
        self.u_max = np.max(max_out) if isinstance(max_out, (list, tuple)) else max_out

        # State for smoothing
        self.prev_voltages = np.zeros(4)
        self.prev_steering = np.zeros(2)
        self.steering_deadband = 0.03
        # Frame handling: if lidar gives global angles, convert to robot-relative
        self.assume_global_angles = config.get('controller', {}).get('fuzzy', {}).get('angles_global', True)

    # ------------------------------------------------------------------
    def action(self, state: np.ndarray, desired_state: np.ndarray, safety_data: Dict[str, Any]) -> np.ndarray:
        """Compute fuzzy-PID style action.

        We assume that the simulation has already collected LIDAR distances and stored
        them in safety_data or config. For simplicity, we look for
        safety_data['lidar_distances'] & safety_data['lidar_angles'].
        If not provided, we issue straight motion.
        """
        distances = safety_data.get('lidar_distances') if safety_data else None
        angles = safety_data.get('lidar_angles') if safety_data else None
        if distances is None or angles is None:
            return self._straight_action()

        distances = np.asarray(distances, dtype=float)
        angles = np.asarray(angles, dtype=float)
        if distances.size == 0:
            return self._straight_action()

        # Current robot yaw (global)
        theta = state[2]

        # Robot-relative beam angles
        rel_angles = np.arctan2(np.sin(angles - theta), np.cos(angles - theta))

        max_range = np.max(distances) if np.max(distances) > 0 else 1.0

        # Goal direction in global frame (from current pose to desired_state position)
        dx = desired_state[0] - state[0]
        dy = desired_state[1] - state[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            # Already at goal: just go straight
            return self._straight_action()
        goal_global = np.arctan2(dy, dx)
        goal_rel = np.arctan2(np.sin(goal_global - theta), np.cos(goal_global - theta))
        goal_rel_deg = float(np.rad2deg(goal_rel))

        # First compute free-space values independent of goal (for potential fallback/info)
        fout_free = self.ff.compute(rel_angles, distances, max_range)
        sec_deg = fout_free['sec_deg']
        sec_free = fout_free['sec_free_val']

        # Compute goal-weighted values using goal_rel_deg
        fout_goal = self.ff.compute(rel_angles, distances, max_range, goal_dir_deg=goal_rel_deg)
        goal_dir_val = fout_goal['goal_dir_val']

        # Select sector
        if np.all(goal_dir_val < 1e-9):
            best_idx = int(np.argmax(sec_free))
            basis = 'sec_free'
        else:
            best_idx = int(np.argmax(goal_dir_val))
            basis = 'goal_dir_val'

        # Desired heading is the selected section center (deg -> rad)
        desired_heading_rel = np.deg2rad(sec_deg[best_idx])
        theta_error = desired_heading_rel  # Already relative
        theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))  # normalize
        if abs(theta_error) < self.cparams.deadband_heading:
            theta_error = 0.0

        # Optional debug logging
        debug = self.config.get('controller', {}).get('fuzzy', {}).get('debug', False)
        # Cache decision
        self.last_decision = {
            'sec_deg': sec_deg.copy(),
            'sec_free': sec_free.copy(),
            'goal_dir_val': goal_dir_val.copy(),
            'goal_rel_deg': goal_rel_deg,
            'basis': basis,
            'best_idx': best_idx,
            'best_deg': sec_deg[best_idx],
            'theta_error_deg': np.rad2deg(theta_error),
        }

        if debug:
            np.set_printoptions(precision=1, suppress=True)
            print(
                f"[Fuzzy] sec_deg={np.round(sec_deg,1)}\n"
                f"         free={np.round(sec_free,2)}\n"
                f"         goal_dir_val={np.round(goal_dir_val,2)} goal_rel_deg={goal_rel_deg:.1f} basis={basis}\n"
                f"         best_idx={best_idx} best_deg={sec_deg[best_idx]:.1f} err_deg={np.rad2deg(theta_error):.2f}"
            )

        # Convert heading error to omega (simple P action on heading)
        omega_cmd = self.cparams.steer_gain * theta_error
        v_long = self.cparams.constant_speed
        v_lat = 0.0  # Not used, but kept for interface similarity

        # Steering calculation (reuse PID logic idea)
        delta_front, delta_rear = self._calculate_steering(v_long, v_lat, omega_cmd, np.hypot(state[3], state[4]))
        # Motor voltages
        voltages = self._calculate_motor_voltages(v_long, delta_front, np.hypot(state[3], state[4]))
        return np.concatenate([[delta_front, delta_rear], voltages])

    # ------------------------------------------------------------------
    def _straight_action(self) -> np.ndarray:
        v = self.cparams.constant_speed
        wheel_radius = self.config['robot']['wheel_radius']
        voltage_needed = v * (self.voltage_speed_factor * self.gear_ratio) / (wheel_radius * self.motor_efficiency + 1e-9)
        voltage_needed = float(np.clip(voltage_needed, -self.v_limit, self.v_limit))
        voltages = np.full(4, voltage_needed)
        voltages = np.clip(voltages, self.u_min, self.u_max)
        voltages = self._smooth_transition(voltages, self.cparams.smoothing_factor)
        return np.concatenate([[0.0, 0.0], voltages])

    # --- Borrowed/Adapted from PID controller ---
    def _calculate_steering(self, v_long, v_lat, omega, vehicle_speed):
        v_long_sign = np.sign(v_long) if abs(v_long) > 0.1 else 1.0
        delta_front = np.arctan2(omega * self.wheelbase + v_lat, v_long_sign * abs(v_long) + 1e-9)
        delta_front = np.clip(delta_front, -self.max_steer_hw, self.max_steer_hw,)
        delta_front = np.clip(delta_front, -self.cparams.max_steer, self.cparams.max_steer)
        delta_rear = self.cparams.front_rear_scale * delta_front
        # Deadband & smoothing similar to PID
        if abs(delta_front) < self.steering_deadband:
            delta_front = 0.0
            delta_rear = 0.0
        elif abs(delta_front - self.prev_steering[0]) < self.steering_deadband:
            delta_front = self.prev_steering[0]
            delta_rear = self.prev_steering[1]
        else:
            smoothing_factor = min(0.6, 0.3 + vehicle_speed / 10.0)
            if np.any(self.prev_steering != 0):
                delta_front = self.prev_steering[0] + smoothing_factor * (delta_front - self.prev_steering[0])
                delta_rear = self.prev_steering[1] + smoothing_factor * (delta_rear - self.prev_steering[1])
        self.prev_steering = np.array([delta_front, delta_rear])
        return float(delta_front), float(delta_rear)

    def _calculate_motor_voltages(self, v_long, delta_front, vehicle_speed):
        V_min = self.config['safety']['limits']['velocity']['min']
        V_max = self.config['safety']['limits']['velocity']['max']
        v_base = np.clip(v_long, V_min, V_max)
        if abs(v_base) < 0.05:
            return np.zeros(4)
        if abs(delta_front) < 0.01:
            wheel_speeds = np.array([v_base] * 4)
        else:
            turn_radius = abs(self.wheelbase / np.tan(delta_front))
            inner_radius = max(0.1, turn_radius - self.track_width / 2)
            outer_radius = turn_radius + self.track_width / 2
            if delta_front > 0:
                wheel_radius = np.array([inner_radius, outer_radius, inner_radius, outer_radius])
            else:
                wheel_radius = np.array([outer_radius, inner_radius, outer_radius, inner_radius])
            wheel_speeds = v_base * wheel_radius / turn_radius
        voltage_speed_factor = self.voltage_speed_factor
        motor_eff = self.motor_efficiency
        motor_voltages = wheel_speeds / (voltage_speed_factor * motor_eff + 1e-9)
        motor_voltages = np.clip(motor_voltages, self.u_min, self.u_max)
        motor_voltages = self._smooth_transition(motor_voltages, self.cparams.smoothing_factor)
        return motor_voltages

    def _smooth_transition(self, new_voltages, smooth_factor=0.1):
        if np.any(self.prev_voltages != 0):
            diff = new_voltages - self.prev_voltages
            for i in range(len(new_voltages)):
                local = smooth_factor * 0.5 if abs(diff[i]) < 0.5 else smooth_factor
                new_voltages[i] = self.prev_voltages[i] + local * diff[i]
        self.prev_voltages = new_voltages.copy()
        return new_voltages
