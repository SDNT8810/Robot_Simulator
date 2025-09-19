import numpy as np
from dataclasses import dataclass
from typing import Tuple


def wrap_to_180(deg: np.ndarray) -> np.ndarray:
    """Wrap angles in degrees to [-180, 180]."""
    wrapped = (deg + 180.0) % 360.0 - 180.0
    # Handle edge case where deg is exactly 180 after wrapping
    wrapped[wrapped == -180.0] = 180.0
    return wrapped


def gaussmf(x: np.ndarray, sigma: float, mean: float) -> np.ndarray:
    """Gaussian membership function (Matlab's gaussmf)."""
    sigma = float(sigma)
    if sigma <= 0:
        # Avoid division by zero; return zero membership if invalid sigma
        return np.zeros_like(x, dtype=float)
    return np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))


def zmf(x: float, ab: Tuple[float, float]) -> float:
    """Z-shaped membership function (approximate Matlab zmf)."""
    a, b = ab
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    t = (x - a) / (b - a)
    return 1.0 - 2.0 * t * t if t < 0.5 else 2.0 * (1.0 - t) ** 2


def smf(x: float, ab: Tuple[float, float]) -> float:
    """S-shaped membership function (approximate Matlab smf)."""
    a, b = ab
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return 2.0 * t * t if t < 0.5 else 1.0 - 2.0 * (1.0 - t) ** 2


@dataclass
class FuzzyParams:
    NL: int = 360              # Number of lidar beams for the angular grid (reference grid)
    Ns: int = 12               # Number of sections
    std_deg: float = 20.0      # Std dev for Gaussian MF in degrees
    G_dirc_deg: float = -160.0 # Goal direction (deg)
    G_STD_deg: float = 70.0    # Std dev for goal direction weighting (deg)


class FuzzyFunctions:
    """
    Compute fuzzy section values from LIDAR beams and provide curves for plotting.
    This mirrors the structure in fuzzy.m using numpy.
    """

    def __init__(self, params: FuzzyParams | None = None):
        self.params = params or FuzzyParams()

    def compute(self, angles_rad: np.ndarray, distances: np.ndarray, max_range: float,
                goal_dir_deg: float | None = None) -> dict:
        """
        Compute fuzzy outputs.

        Inputs:
          - angles_rad: beam angles (rad), shape (N,)
          - distances: beam distances (m), shape (N,)
          - max_range: lidar max range (m)
          - goal_dir_deg: optional goal direction in degrees; if None, uses params.G_dirc_deg

        Returns a dict with keys:
          - deg_grid: degrees for each beam (length N)
          - lidar_norm: distances normalized to [0,1] by max_range
          - sec_deg: section center degrees (length Ns)
          - sec_free_val: free value per section in [0,1]
          - goal_dir_val: weighted section values (by goal direction)
          - goal_membership: tuple (mu_close, mu_near, mu_far)
        """
        p = self.params
        # Convert angles to degrees and wrap
        degs = np.rad2deg(angles_rad).astype(float)
        degs = wrap_to_180(degs)

        # Normalize distances
        Xmax = float(max_range) if max_range and max_range > 0 else 1.0
        lidar_norm = np.clip(distances / Xmax, 0.0, 1.0)

        # Sections
        Ns = int(p.Ns)
        # Centers from -180 to 180 over Ns; mimic fuzzy.m where first value drops -180
        sec_deg = np.linspace(-180.0, 180.0, Ns + 1)
        sec_deg = sec_deg[1:]  # remove -180 to match MATLAB pattern

        # Compute membership weights W2 for each section over all beams
        # For each section i, weight depends on angular distance from section center
        std = float(p.std_deg)
        W2 = np.zeros((Ns, degs.shape[0]), dtype=float)
        for i in range(Ns):
            center = sec_deg[i]
            diff = wrap_to_180(degs - center)
            W2[i, :] = gaussmf(diff, std, 0.0)

        # Weighted average of lidar beams across sections
        # Use original (not symmetrized) beams as in MATLAB: Mu = W2(i,:)*Lidar_Beams/Xmax; Sec_Free_Val = Mu/sum(W2)
        # lidar_norm already divided by Xmax
        sec_free_val = np.zeros(Ns, dtype=float)
        weight_sums = W2.sum(axis=1) + 1e-9
        for i in range(Ns):
            Mu = float(W2[i, :].dot(lidar_norm))
            sec_free_val[i] = Mu / weight_sums[i]

        # Goal direction consideration
        G_dir = float(p.G_dirc_deg if goal_dir_deg is None else goal_dir_deg)
        G_STD = float(p.G_STD_deg)
        wgj = gaussmf(sec_deg, G_STD, G_dir)
        goal_dir_val = wgj * sec_free_val

        # Optional goal distance MFs (example: use average 1 - lidar_norm as proxy for closeness)
        # In MATLAB fuzzy.m, G_dist is a scalar; here we derive a simple scalar distance indicator
        # based on min distance but clamp to [0, 5] for membership functions.
        G_dist = float(np.clip(np.minimum(5.0, np.maximum(0.0, Xmax * (1.0 - np.max(lidar_norm)))), 0.0, 5.0))
        mu_close = zmf(G_dist, (0.0, 1.0))
        mu_near = float(np.mean(gaussmf(np.array([G_dist]), 1.0, 2.0)))
        mu_far = smf(G_dist, (2.0, 5.0))

        return {
            'deg_grid': degs,
            'lidar_norm': lidar_norm,
            'sec_deg': sec_deg,
            'sec_free_val': sec_free_val,
            'goal_dir_val': goal_dir_val,
            'goal_membership': (mu_close, mu_near, mu_far),
        }
