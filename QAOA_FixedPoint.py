
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import product


# 1. Convert graph to Boolean network
def convert_to_boolean_graph(edges, vertices):
    neighbors = {v: [] for v in vertices}

    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)

    return neighbors


# 2. Boolean update function
def boolean_update(state, vertices, neighbors):
    """
    Example rule:
    Each vertex becomes 1 if any neighbor is 1.
    An isolated vertex becomes 0.
    """
    state_map = dict(zip(vertices, state))

    return tuple(
        int(any(state_map[u] for u in neighbors[v]))
        for v in vertices
    )

#Add "and" here.
#Synchronous update: all vertices update simultaneously based on the current state. <- add
#Non determinisitic update: introduce randomness in the update rule, e.g., a vertex may flip its state with some probability. <- add

# 3. Find all fixed points
def get_all_fixed_points(edges, vertices):
    neighbors = convert_to_boolean_graph(edges, vertices)
    n = len(vertices)

    states = list(product([0, 1], repeat=n))

    costs = []
    vals = []
    fixed_points = []

    for state in states:
        next_state = boolean_update(
            state, vertices, neighbors
        )

        # Number of satisfied fixed-point equations
        #Rework this to understand better
        #Opposite minimizing
        cost = sum(
            int(state[i] == next_state[i])
            for i in range(n)
        )

        bitstring = "".join(map(str, state))

        costs.append(cost)
        vals.append(bitstring)

        if state == next_state:
            fixed_points.append(bitstring)

    return np.array(costs), vals, fixed_points


# 4. QAOA evolution
def run_fixed_point_QAOA(
    gamma, beta, costs, num_layers
):
    n_states = len(costs)
    n = int(np.log2(n_states))

    # Uniform superposition
    amplitudes = np.ones(
        n_states, dtype=complex
    ) / np.sqrt(n_states)

    # Cost unitary
    Uc = np.exp(-1j * gamma * costs)

    # Mixer unitary
    bitstrings = [
        format(i, f"0{n}b")
        for i in range(n_states)
    ]

    Um = np.zeros(
        (n_states, n_states), dtype=complex
    )

    for i, x in enumerate(bitstrings):
        for j, y in enumerate(bitstrings):
            d = sum(a != b for a, b in zip(x, y))

            Um[i, j] = (
                np.cos(beta)**(n-d)
                * (-1j*np.sin(beta))**d
            )

    for _ in range(num_layers):
        amplitudes = Um @ (Uc * amplitudes)

    return amplitudes


# 5. Optimize gamma, beta, layers
def optimize_fixed_points(
    edges,
    vertices,
    n_gamma=20,
    n_beta=20,
    max_runs=30
):
    costs, vals, fixed_points = get_all_fixed_points(
        edges, vertices
    )
    #change cost function to include AND, OR, XOR, etc. based on the network structure
    print(costs)

    if not fixed_points:
        raise ValueError(
            "The Boolean network has no fixed points."
        )

    n = len(vertices)

    # Fixed point mask
    fixed_mask = costs == n

    gammas = np.linspace(0, 2*np.pi, n_gamma)
    betas = np.linspace(0, 2*np.pi, n_beta)
    runs = np.arange(1, max_runs+1)

    # gamma x beta x layers
    probabilities = np.zeros((
        len(gammas),
        len(betas),
        len(runs)
    ))

    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            for k, num_runs in enumerate(runs):

                amplitudes = run_fixed_point_QAOA(
                    gamma, beta,
                    costs, int(num_runs)
                )

                probabilities[i, j, k] = np.sum(
                    np.abs(amplitudes[fixed_mask])**2
                )

    # Best parameter combination
    best_i, best_j, best_k = np.unravel_index(
        np.argmax(probabilities),
        probabilities.shape
    )

    best_gamma = gammas[best_i]
    best_beta = betas[best_j]
    best_runs = runs[best_k]
    best_probability = probabilities[
        best_i, best_j, best_k
    ]

    print("Boolean network fixed points:", fixed_points)
    print(f"Best gamma: {best_gamma:.6f}")
    print(f"Best beta: {best_beta:.6f}")
    print(f"Best number of layers: {best_runs}")
    print(f"Maximum probability: {best_probability:.6f}")

    # --------------------------------
    # Interactive heatmap
    # --------------------------------

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.22)

    im = ax.imshow(
        probabilities[:, :, 0].T,
        origin="lower",
        aspect="auto",
        extent=[
            gammas[0], gammas[-1],
            betas[0], betas[-1]
        ],
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    fig.colorbar(
        im, ax=ax,
        label="Probability of Measuring a Fixed Point"
    )

    ax.set_xlabel("Gamma (γ)")
    ax.set_ylabel("Beta (β)")

    angle_ticks = [
        0, np.pi/2, np.pi,
        3*np.pi/2, 2*np.pi
    ]
    angle_labels = [
        "0", "π/2", "π", "3π/2", "2π"
    ]

    ax.set_xticks(angle_ticks, angle_labels)
    ax.set_yticks(angle_ticks, angle_labels)

    marker, = ax.plot(
        [], [], "r*",
        markersize=15,
        markeredgecolor="white"
    )

    def update(run_value):
        k = int(run_value) - 1

        im.set_data(probabilities[:, :, k].T)

        i, j = np.unravel_index(
            np.argmax(probabilities[:, :, k]),
            probabilities[:, :, k].shape
        )

        marker.set_data(
            [gammas[i]],
            [betas[j]]
        )

        ax.set_title(
            f"Fixed-Point QAOA | Layers = {runs[k]}"
            f" | Max P = {probabilities[i,j,k]:.4f}"
        )

        fig.canvas.draw_idle()

    slider_ax = fig.add_axes(
        [0.2, 0.08, 0.65, 0.03]
    )

    run_slider = Slider(
        slider_ax,
        "Layers",
        valmin=1,
        valmax=max_runs,
        valinit=1,
        valstep=1
    )

    run_slider.on_changed(update)
    update(1)

    plt.show()

    # --------------------------------
    # Best probability vs layers
    # --------------------------------

    best_per_run = probabilities.max(axis=(0, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(runs, best_per_run, linewidth=2)

    plt.xlabel("Number of QAOA Layers")
    plt.ylabel("Maximum Fixed-Point Probability")
    plt.title("Fixed-Point QAOA Optimization")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Final state probabilities at best parameters
    best_amplitudes = run_fixed_point_QAOA(
        best_gamma, best_beta, costs, best_runs
    )

    print("\nState probabilities:")
    for state, amp in zip(vals, best_amplitudes):
        print(
            f"{state}: {abs(amp)**2:.6f}"
            f"{'  <-- fixed point' if state in fixed_points else ''}"
        )

    return {
        "gammas": gammas,
        "betas": betas,
        "runs": runs,
        "probabilities": probabilities,
        "fixed_points": fixed_points,
        "best_gamma": best_gamma,
        "best_beta": best_beta,
        "best_runs": best_runs,
        "best_probability": best_probability
    }


# ------------------------------------
# Example
# ------------------------------------

vertices = [1, 2, 3]
edges = [(1, 2), (2, 3), (3, 1)]

results = optimize_fixed_points(
    edges,
    vertices,
    n_gamma=10,
    n_beta=10,
    max_runs=20
)

#New network:
# (1, 2), (2, 3), (3, 1)
# Each had 3 inputs (and, or, and)
# WTF 6x6x6: 1, and, x, y, or, 0
# Add input state...
# Base of attraction <- look up paper on this.

#Convert to qiskit