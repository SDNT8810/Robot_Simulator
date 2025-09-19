import numpy as np
import matplotlib.pyplot as plt

class Lidar:
    """
    Simulate and visualize LIDAR beams for a robot.

    Obstacles are modeled as circles: (cx, cy, r).
    """
    def __init__(self, num_beams=32, max_range=6.0, fov=np.pi*2):
        self.num_beams = int(num_beams)
        self.max_range = float(max_range)
        self.fov = float(fov)

    def _ray_circle_intersection(self, ox, oy, dx, dy, cx, cy, r):
        """
        Compute the smallest non-negative intersection distance t along the ray
        p(t) = o + t d, with circle centered at c with radius r. Returns None if no hit.
        Assumes direction d is unit length.
        """
        ocx, ocy = ox - cx, oy - cy
        b = 2.0 * (dx * ocx + dy * ocy)
        c = (ocx * ocx + ocy * ocy) - r * r
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        hits = [t for t in (t1, t2) if t >= 0.0]
        if not hits:
            return None
        return min(hits)

    def cast_rays(self, x, y, theta, obstacles):
        """
        Cast rays and clip them by circular obstacles.

        Args:
            x, y, theta: robot pose
            obstacles: iterable of (cx, cy, r)

        Returns:
            endpoints: (N, 2) array of clipped endpoints
            distances: (N,) array of distances to hit (max_range if no hit)
            angles: (N,) array of absolute beam angles
        """
        angles = np.linspace(-self.fov / 2.0, self.fov / 2.0, self.num_beams) + theta
        endpoints = np.zeros((self.num_beams, 2), dtype=float)
        distances = np.zeros(self.num_beams, dtype=float)

        ox, oy = float(x), float(y)

        for i, a in enumerate(angles):
            dx, dy = np.cos(a), np.sin(a)
            t_min = self.max_range
            # Check intersections with all circular obstacles
            for (cx, cy, r) in obstacles or []:
                t_hit = self._ray_circle_intersection(ox, oy, dx, dy, float(cx), float(cy), float(r))
                if t_hit is not None and t_hit < t_min:
                    t_min = t_hit
            # Endpoint after clipping by nearest obstacle or max_range
            ex, ey = ox + t_min * dx, oy + t_min * dy
            endpoints[i, 0], endpoints[i, 1] = ex, ey
            distances[i] = t_min

        return endpoints, distances, angles

    def plot_beams(self, ax, x, y, theta, obstacles=None, color='orange', alpha=0.5, linewidth=1, show_hits=False):
        """
        Plot LIDAR beams clipped by obstacles on the given axes.
        """
        endpoints, distances, angles = self.cast_rays(x, y, theta, obstacles or [])
        for end in endpoints:
            ax.plot([x, end[0]], [y, end[1]], color=color, alpha=alpha, linewidth=linewidth)
        if show_hits:
            ax.scatter(endpoints[:, 0], endpoints[:, 1], c=color, s=6, alpha=alpha)
