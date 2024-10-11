import numpy as np
from filterpy.kalman import KalmanFilter
from simulate import WaterDisseminationSimulation
import networkx as nx
import random
import matplotlib.pyplot as plt


class NetworkInferenceAndFlowEstimator:
    def __init__(
        self,
        n_nodes,
        dt,
        process_noise_std,
        measurement_noise_std,
        correlation_threshold,
    ):
        self.n_nodes = n_nodes
        self.correlation_threshold = correlation_threshold
        self.kf = KalmanFilter(dim_x=n_nodes, dim_z=n_nodes)
        self.kf.F = self.kf.H = np.eye(n_nodes)
        self.kf.R = np.eye(n_nodes) * measurement_noise_std**2
        self.kf.Q = np.eye(n_nodes) * process_noise_std**2 * dt
        self.kf.P = np.eye(n_nodes) * 100

    def infer_network(self, node_history):
        node_array = np.array(
            [[step[node] for node in range(self.n_nodes)] for step in node_history]
        )
        correlation_matrix = np.corrcoef(node_array.T)
        self.inferred_network = nx.Graph(
            [
                (i, j)
                for i in range(self.n_nodes)
                for j in range(i + 1, self.n_nodes)
                if abs(correlation_matrix[i, j]) > self.correlation_threshold
            ]
        )

    def update(self, measurements):
        self.kf.predict()
        self.kf.update(measurements)

    def estimate_flows(self, dissemination_speed):
        return {
            edge: abs(dissemination_speed * (self.kf.x[edge[0]] - self.kf.x[edge[1]]))
            for edge in self.inferred_network.edges()
        }


def compare_networks(actual_network, inferred_network):
    actual_edges = set(actual_network.edges())
    inferred_edges = set(inferred_network.edges())
    tp = len(actual_edges.intersection(inferred_edges))
    fp = len(inferred_edges - actual_edges)
    fn = len(actual_edges - inferred_edges)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1_score


def compare_flows(actual_flows, estimated_flows):
    common_edges = set(actual_flows.keys()) & set(estimated_flows.keys())
    return (
        np.mean(
            [(actual_flows[edge] - estimated_flows[edge]) ** 2 for edge in common_edges]
        )
        if common_edges
        else np.inf
    )


def plot_mse(mse_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mse_history) + 1), mse_history)
    plt.xlabel("Time Step")
    plt.ylabel("Mean Squared Error")
    plt.title("Flow Estimation Error Over Time")
    plt.show()


def plot_flow_comparison(common_edges, actual_flows_history, estimated_flows_history):
    num_edges = len(common_edges)
    num_cols = 3
    num_rows = (num_edges + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 2 * num_rows))
    fig.suptitle("Flow Comparison for Common Edges", fontsize=16)

    for i, edge in enumerate(common_edges):
        ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i % num_cols]
        actual_flow = [flows.get(edge, 0) for flows in actual_flows_history]
        estimated_flow = [flows.get(edge, 0) for flows in estimated_flows_history]

        ax.plot(range(1, len(actual_flow) + 1), actual_flow, label="Actual Flow")
        ax.plot(
            range(1, len(estimated_flow) + 1), estimated_flow, label="Estimated Flow"
        )
        ax.set_title(f"Edge {edge}", fontsize=10)
        ax.set_xlabel("Time Step", fontsize=8)
        ax.set_ylabel("Flow", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.legend(fontsize=6)

    for i in range(num_edges, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_networks(actual_network, inferred_network):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pos = nx.spring_layout(actual_network)

    nx.draw(
        actual_network,
        pos,
        ax=ax1,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
    )
    ax1.set_title("Actual Network")

    nx.draw(
        inferred_network,
        pos,
        ax=ax2,
        with_labels=True,
        node_color="lightgreen",
        node_size=500,
        font_size=10,
        font_weight="bold",
    )
    ax2.set_title("Inferred Network")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    (
        n_nodes,
        initial_water_node,
        num_steps,
        dissemination_speed,
        dt,
        correlation_threshold,
    ) = (20, 0, 500, 0.1, 1.0, 0.5)
    network_types = {
        "erdos_renyi": {"p": 0.1},
        "barabasi_albert": {"m": 2},
        "watts_strogatz": {"k": 4, "p": 0.1},
    }
    network_type = random.choice(list(network_types.keys()))

    print(f"Simulating with network type: {network_type}")
    sim = WaterDisseminationSimulation(
        network_type,
        n_nodes,
        initial_water_node,
        dissemination_speed,
        **network_types[network_type],
    )
    sim.simulate(num_steps)

    estimator = NetworkInferenceAndFlowEstimator(
        n_nodes, dt, 0.1, 0.1, correlation_threshold
    )
    estimator.infer_network(sim.get_node_history())

    precision, recall, f1_score = compare_networks(
        sim.get_network(), estimator.inferred_network
    )
    print(
        f"Network Structure Inference: Precision={precision:.2f}, Recall={recall:.2f}, F1-score={f1_score:.2f}"
    )

    estimated_flows_history, mse_history, actual_flows_history = [], [], []
    for node_levels in sim.get_node_history()[1:]:
        estimator.update(np.array(list(node_levels.values())))
        estimated_flows = estimator.estimate_flows(dissemination_speed)
        estimated_flows_history.append(estimated_flows)
        actual_flows = sim.get_edge_history()[len(estimated_flows_history)]
        actual_flows_history.append(actual_flows)
        mse_history.append(compare_flows(actual_flows, estimated_flows))

    plot_mse(mse_history)

    common_edges = set(sim.get_network().edges()) & set(
        estimator.inferred_network.edges()
    )
    if common_edges:
        plot_flow_comparison(
            common_edges, actual_flows_history, estimated_flows_history
        )
    else:
        print("No common edges found between actual and inferred networks.")

    plot_networks(sim.get_network(), estimator.inferred_network)
