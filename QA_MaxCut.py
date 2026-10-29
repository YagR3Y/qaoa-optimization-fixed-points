#Fixed point proj example
#QAOA 

#Example graph
# 2 - 3
# |
# 1
from matplotlib import pyplot as plt
from matplotlib.pylab import beta, full
from numpy import cos, exp, linspace, nan, pi, sin, sqrt, zeros
import numpy as np


vertices = [1,2,3]
edges = [(1,2), (2,3)]

def get_all_cuts(edges, vertices):
    n = len(vertices)
    combos = []
    all_cost = []
    all_vals = []
    for i in range(2**n):
        combo = []
        cost = 0
        for j in range(n):
            if i & (1 << j):
                combo.append(vertices[j])
        cut = []
        for u, v in edges:
            if (u in combo) != (v in combo):
                cut.append((u, v))
                cost += 1
                #Return 1 if edge is cut, 0 otherwise
        combos.append((combo, cut))
        all_cost.append(cost)
    for combo, cut in combos:
        binary = convert_combo_to_binary(combo, vertices)
        all_vals.append(binary)
    return all_cost, all_vals

def convert_combo_to_binary(combo, vertices):
    binary = ""
    for v in vertices:
        if v in combo:
            binary += "1"
        else:
            binary += "0"
    return binary
#C(x) = (x1+x2-x1*x2) + (x2+x3-x2*x3)
#H_C()|xxx> = C(x)|xxx>

#Unitary cost
#U_C(γ) = exp(−iγH_C)
def U_C(gamma, Hc):
    uni_cost = []
    for i in range(len(Hc)):
        val = exp(-1j * gamma * Hc[i])
        uni_cost.append(val)
    return uni_cost

#Mixer
def Hamming_dist_binary(x, y):
    dist = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            dist += 1
    return dist

#U_M(β) = e^(−iβH_M)
#⟨x|U_M(β)|y⟩ = cosβ ^(n−d) (−isinβ)^d
#where n is the number of qubits and d is the Hamming distance between x and y
def U_M(beta, Ham_dist, n):
    return cos(beta)**(n-Ham_dist) * (-1j*sin(beta))**Ham_dist
#Return


def QAOA(gamma, beta, edges, vertices, initial_amplitudes):
    costs, vals = get_all_cuts(edges, vertices)
    n = len(vertices)
    Uc = U_C(gamma, costs)
    all_amplitudes = []
    for outcome in vals:
        a = 0
        for j, val in enumerate(vals):
            Ham_dist = Hamming_dist_binary(val, outcome)
            Um = U_M(beta, Ham_dist, n)
            amplitude = initial_amplitudes[j] * Uc[j] * Um
            a += amplitude

        all_amplitudes.append(a)
    return all_amplitudes


def run_QAOA_multiple_times(gamma, beta, edges, vertices, num_runs):
    n = len(vertices)
    amplitudes = [1/sqrt(2**n) for _ in range(2**n)]
    for _ in range(num_runs):
        amplitudes = QAOA(gamma, beta, edges, vertices, amplitudes)

    return amplitudes

#1
#Optimize the parameters gamma and beta to maximize the probability of the optimal cut
def optimize_parameters(edges, vertices, num_runs):
    costs, vals = get_all_cuts(edges, vertices)
    optimal_cost = max(costs)
    best_gamma = None
    best_beta = None
    best_probability = 0

    for gamma in [i * pi / 20 for i in range(21)]:
        for beta in [i * pi / 20 for i in range(21)]:
            amplitudes = run_QAOA_multiple_times(gamma, beta, edges, vertices, num_runs)
            probability = sum(abs(a)**2 for a, c in zip(amplitudes, costs) if c == optimal_cost)

            if probability > best_probability:
                best_probability = probability
                best_gamma = gamma
                best_beta = beta

    return best_gamma, best_beta, best_probability

#Create graph of gamma and beta values and their corresponding probabilities
def graph_QAOA_parameters(edges, vertices, num_layers=1):
    costs, vals = get_all_cuts(edges, vertices)

    gammas = linspace(0, 2*pi, 100)
    betas = linspace(0, 2*pi, 100)

    probabilities = zeros((len(betas), len(gammas)))
    max_cost = max(costs)

    for i, beta in enumerate(betas):
        for j, gamma in enumerate(gammas):

            amplitudes = run_QAOA_multiple_times(
                gamma, beta, edges, vertices, num_layers
            )

            # Sum probabilities of optimal solutions
            probabilities[i, j] = sum(
                abs(amp)**2
                for amp, cost in zip(amplitudes, costs)
                if cost == max_cost
            )

    plt.figure(figsize=(10, 7))
    plt.imshow(
        probabilities,
        extent=[gammas[0], gammas[-1],
                betas[0], betas[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis"
    )

    plt.colorbar(label="Probability of Optimal Solution")
    plt.xlabel("Gamma (γ)")
    plt.ylabel("Beta (β)")
    plt.title("QAOA Optimal Solution Probability")
    plt.show()

def vary_beta_only(edges, vertices, gamma, num_layers):
    costs, vals = get_all_cuts(edges, vertices)
    max_cost = max(costs)

    betas = linspace(0, 2*pi, 100)
    probabilities = []

    for beta in betas:
        amplitudes = run_QAOA_multiple_times(
            gamma, beta, edges, vertices, num_layers
        )

        probability = sum(
            abs(amp)**2
            for amp, cost in zip(amplitudes, costs)
            if cost == max_cost
        )
        probabilities.append(probability)

    plt.figure(figsize=(10, 5))
    plt.plot(betas, probabilities, marker="o")
    plt.xlabel("Beta (β)")
    plt.ylabel("Probability of Optimal Solution")
    plt.title(f"Varying Beta (γ={gamma:.3f}, Layers={num_layers})")
    plt.grid(alpha=0.3)
    plt.show()    

#2
#Optimize the number of iterations to maximize the probability of the optimal cut
def optimize_runs(edges, vertices, gamma, beta):
    costs, vals = get_all_cuts(edges, vertices)
    optimal_cost = max(costs)

    best_probability = -1
    best_num_runs = 0

    runs = []
    probabilities = []

    for num_runs in range(1, 100, 1):
        amplitudes = run_QAOA_multiple_times(
            gamma, beta, edges, vertices, num_runs
        )

        probability = sum(
            abs(a)**2
            for a, c in zip(amplitudes, costs)
            if c == optimal_cost
        )

        runs.append(num_runs)
        probabilities.append(probability)

        if probability > best_probability:
            best_probability = probability
            best_num_runs = num_runs

    # Graph results
    plt.figure(figsize=(10, 5))
    plt.plot(runs, probabilities, marker="o", label="Optimal Cut Probability")

    # Highlight best result
    plt.scatter(
        best_num_runs,
        best_probability,
        color="red",
        s=100,
        zorder=5,
        label=f"Best: {best_num_runs} layers"
    )

    plt.xlabel("Number of QAOA Layers")
    plt.ylabel("Probability of Optimal Cut")
    plt.title(f"QAOA Optimization (γ={gamma:.3f}, β={beta:.3f})")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Best number of layers: {best_num_runs}")
    print(f"Maximum probability: {best_probability:.6f}")

    return best_num_runs

def optimize_runs_while_varying_beta(edges, vertices, gamma):
    costs, vals = get_all_cuts(edges, vertices)
    optimal_cost = max(costs)

    betas = np.linspace(0, 2 * np.pi, 20)
    num_runs_values = np.arange(1, 100)

    # Rows = beta, columns = number of runs
    probabilities = np.zeros((len(betas), len(num_runs_values)))

    for i, beta in enumerate(betas):
        for j, num_runs in enumerate(num_runs_values):
            amplitudes = run_QAOA_multiple_times(
                gamma, beta, edges, vertices, num_runs
            )

            probabilities[i, j] = sum(
                abs(a)**2
                for a, c in zip(amplitudes, costs)
                if c == optimal_cost
            )

    # Find the maximum
    best_i, best_j = np.unravel_index(
        np.argmax(probabilities),
        probabilities.shape
    )

    best_beta = betas[best_i]
    best_num_runs = num_runs_values[best_j]
    best_probability = probabilities[best_i, best_j]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    mesh = ax.pcolormesh(
        num_runs_values,
        betas,
        probabilities,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Probability of Optimal Cut"
    )

    # Highlight the best parameters
    ax.scatter(
        best_num_runs,
        best_beta,
        color="red",
        marker="*",
        s=250,
        edgecolors="white",
        linewidths=0.8,
        label=f"Best: runs={best_num_runs}, β={best_beta:.3f}"
    )

    ax.set_xlabel("Number of QAOA Runs")
    ax.set_ylabel("Beta (β)")
    ax.set_title(f"QAOA Optimization Heatmap (γ={gamma:.3f})")

    ax.set_yticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        ["0", "π/2", "π", "3π/2", "2π"]
    )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    print(f"Best number of runs: {best_num_runs}")
    print(f"Best beta: {best_beta:.6f}")
    print(f"Maximum probability: {best_probability:.6f}")

    return betas, num_runs_values, probabilities


import numpy as np
import matplotlib.pyplot as plt

def optimize_runs_while_varying_gamma(edges, vertices, beta):
    costs, vals = get_all_cuts(edges, vertices)
    optimal_cost = max(costs)

    gammas = np.linspace(0, 2 * np.pi, 20)
    num_runs_values = np.arange(1, 100)

    # Rows = gamma, columns = number of runs
    probabilities = np.zeros((len(gammas), len(num_runs_values)))

    for i, gamma in enumerate(gammas):
        for j, num_runs in enumerate(num_runs_values):
            amplitudes = run_QAOA_multiple_times(
                gamma, beta, edges, vertices, num_runs
            )

            probabilities[i, j] = sum(
                abs(a)**2
                for a, c in zip(amplitudes, costs)
                if c == optimal_cost
            )

    # Find the maximum probability
    best_i, best_j = np.unravel_index(
        np.argmax(probabilities),
        probabilities.shape
    )

    best_gamma = gammas[best_i]
    best_num_runs = num_runs_values[best_j]
    best_probability = probabilities[best_i, best_j]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    mesh = ax.pcolormesh(
        num_runs_values,
        gammas,
        probabilities,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Probability of Optimal Cut"
    )

    # Highlight best parameters
    ax.scatter(
        best_num_runs,
        best_gamma,
        color="red",
        marker="*",
        s=250,
        edgecolors="white",
        linewidths=0.8,
        label=f"Best: runs={best_num_runs}, γ={best_gamma:.3f}"
    )

    ax.set_xlabel("Number of QAOA Runs")
    ax.set_ylabel("Gamma (γ)")
    ax.set_title(f"QAOA Optimization Heatmap (β={beta:.3f})")

    ax.set_yticks(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        ["0", "π/2", "π", "3π/2", "2π"]
    )

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    print(f"Best number of runs: {best_num_runs}")
    print(f"Best gamma: {best_gamma:.6f}")
    print(f"Maximum probability: {best_probability:.6f}")

    return gammas, num_runs_values, probabilities


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def optimize_runs_varying_gamma_beta(
    edges, vertices,
    n_gamma=20,
    n_beta=20,
    max_runs=99
):
    costs, vals = get_all_cuts(edges, vertices)
    costs = np.asarray(costs)
    optimal_cost = np.max(costs)

    optimal_mask = (costs == optimal_cost)

    # Parameter ranges
    gammas = np.linspace(0, 2 * np.pi, n_gamma)
    betas = np.linspace(0, 2 * np.pi, n_beta)
    runs = np.arange(1, max_runs + 1)

    # Shape: (gamma, beta, num_runs)
    probabilities = np.zeros((
        len(gammas),
        len(betas),
        len(runs)
    ))

    # Evaluate every parameter combination
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            for k, num_runs in enumerate(runs):

                amplitudes = np.asarray(
                    run_QAOA_multiple_times(
                        gamma, beta,
                        edges, vertices, int(num_runs)
                    )
                )

                probabilities[i, j, k] = np.sum(
                    np.abs(amplitudes[optimal_mask])**2
                )

    # Find global maximum
    best_i, best_j, best_k = np.unravel_index(
        np.argmax(probabilities),
        probabilities.shape
    )

    best_gamma = gammas[best_i]
    best_beta = betas[best_j]
    best_runs = runs[best_k]
    best_probability = probabilities[best_i, best_j, best_k]

    print(f"Best gamma: {best_gamma:.6f}")
    print(f"Best beta: {best_beta:.6f}")
    print(f"Best number of runs: {best_runs}")
    print(f"Maximum probability: {best_probability:.6f}")

    # Interactive heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.22)

    initial_k = 0

    im = ax.imshow(
        probabilities[:, :, initial_k].T,
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
        label="Probability of Optimal Cut"
    )

    ax.set_xlabel("Gamma (γ)")
    ax.set_ylabel("Beta (β)")
    ax.set_title(f"QAOA Probability | Runs = {runs[initial_k]}")

    angle_ticks = [
        0, np.pi / 2, np.pi,
        3 * np.pi / 2, 2 * np.pi
    ]
    angle_labels = [
        "0", "π/2", "π", "3π/2", "2π"
    ]

    ax.set_xticks(angle_ticks, angle_labels)
    ax.set_yticks(angle_ticks, angle_labels)

    # Mark the best gamma/beta for selected run count
    marker, = ax.plot(
        [], [],
        "r*", markersize=15,
        markeredgecolor="white"
    )

    def update(run_value):
        k = int(run_value) - 1

        im.set_data(probabilities[:, :, k].T)

        # Best gamma and beta at this run count
        i, j = np.unravel_index(
            np.argmax(probabilities[:, :, k]),
            probabilities[:, :, k].shape
        )

        marker.set_data(
            [gammas[i]],
            [betas[j]]
        )

        ax.set_title(
            f"QAOA Probability | Runs = {runs[k]} "
            f"| Max P = {probabilities[i, j, k]:.4f}"
        )

        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.2, 0.08, 0.65, 0.03])

    run_slider = Slider(
        slider_ax,
        "Runs",
        valmin=1,
        valmax=max_runs,
        valinit=1,
        valstep=1
    )

    run_slider.on_changed(update)
    update(1)

    plt.show()

    return {
        "gammas": gammas,
        "betas": betas,
        "runs": runs,
        "probabilities": probabilities,
        "best_gamma": best_gamma,
        "best_beta": best_beta,
        "best_runs": best_runs,
        "best_probability": best_probability
    }

def run_all_optimizations(edges, vertices):
    results = optimize_runs_varying_gamma_beta(
    edges,
    vertices,
    n_gamma=10,
    n_beta=10,
    max_runs=20
    )

    probabilities = results["probabilities"]
    runs = results["runs"]

    best_per_run = probabilities.max(axis=(0, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(runs, best_per_run, linewidth=2)

    plt.xlabel("Number of QAOA Runs")
    plt.ylabel("Maximum Probability of Optimal Cut")
    plt.title("Best QAOA Probability vs. Number of Runs")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    costs, vals = get_all_cuts(edges, vertices)
    amplitudes = run_QAOA_multiple_times(1.2566370614359172, 0.47123889803846897, edges, vertices, 51)
    for val, cost, amp in zip(vals, costs, amplitudes):
        probability = abs(amp)**2
        print(
            f"State: {val}, "
            f"Cost: {cost}, "
            f"Probability: {probability:.6f}"
        )
    print("Total probability:", sum(abs(a)**2 for a in amplitudes))
    #best_gamma, best_beta, best_probability = optimize_parameters(edges, vertices, 1)
    #print(f"Best gamma: {best_gamma}, Best beta: {best_beta}, Best probability: {best_probability}")
    #vary_beta_only(edges, vertices, 1.2566370614359172, 1)
    #graph_QAOA_parameters(edges, vertices, num_layers=1)

    #Run 1 at a time
    optimize_runs_while_varying_beta(edges, vertices, 1.2566370614359172)
    optimize_runs_while_varying_gamma(edges, vertices, 0.47123889803846897)
    
    #Does what you think
    #run_all_optimizations(edges, vertices)



main()

#TODO:
# Sampling at higher iterations for run
# Change beta and sample again
# Is Y or B changing the frequency of the num of layers?
# Switch up betas/gammas per run
# Fix varying beta/gamma per run
# Optimize beta/gamma per run