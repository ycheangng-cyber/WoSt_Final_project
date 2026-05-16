import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Signed Distance Function (unit circle)
# =========================================================

def sdf_circle(p):
    return 1.0 - np.linalg.norm(p)


# =========================================================
# Boundary condition
# =========================================================

def boundary_value(p):
    x, y = p
    return x**2 - y**2


# =========================================================
# Uniform direction sampling on unit circle
# =========================================================

def sample_direction():
    theta = 2.0 * np.pi * np.random.rand()
    return np.array([
        np.cos(theta),
        np.sin(theta)
    ])


# =========================================================
# Single walk
# =========================================================

def walk_on_stars(
    x0,
    eps=1e-4,
    max_steps=1000
):
    x = np.array(x0, dtype=np.float64)

    for _ in range(max_steps):

        # distance to boundary
        r = sdf_circle(x)

        # close enough to boundary
        if r < eps:
            x_boundary = x / np.linalg.norm(x)
            return boundary_value(x_boundary)

        # random direction
        w = sample_direction()

        # WoS step
        x = x + r * w

    return 0.0


# =========================================================
# Monte Carlo estimator
# =========================================================

def solve_point(
    x0,
    n_walks=10000
):
    vals = [walk_on_stars(x0) for _ in range(n_walks)]
    return np.mean(vals)


# =========================================================
# Solve on grid
# =========================================================

def solve_grid(
    resolution=80,
    n_walks=2000
):

    xs = np.linspace(-1, 1, resolution)
    ys = np.linspace(-1, 1, resolution)

    U = np.full((resolution, resolution), np.nan)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            p = np.array([x, y])

            if np.linalg.norm(p) < 1.0:
                U[j, i] = solve_point(p, n_walks)

    return xs, ys, U


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    xs, ys, U = solve_grid(
        resolution=60,
        n_walks=1000
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(
        U,
        extent=[-1, 1, -1, 1],
        origin='lower'
    )

    plt.colorbar(label='u(x,y)')
    plt.title('Walk-on-Stars Laplace Solver')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()