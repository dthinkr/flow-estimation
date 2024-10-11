import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class WaterDisseminationSimulation:
    def __init__(
        self,
        network_type,
        n_nodes,
        initial_water_node,
        dissemination_speed,
        **network_params,
    ):
        self.G = getattr(nx, f"{network_type}_graph")(n_nodes, **network_params)
        self.initial_water_node = initial_water_node
        self.dissemination_speed = dissemination_speed
        self.node_history = []
        self.edge_history = []

    def initialize_water(self, initial_water_level=1.0):
        nx.set_node_attributes(self.G, 0, "water_level")
        self.G.nodes[self.initial_water_node]["water_level"] = initial_water_level
        nx.set_edge_attributes(self.G, 0, "water_flow")

    def update_network(self):
        new_levels = {
            node: self.G.nodes[node]["water_level"] for node in self.G.nodes()
        }
        for node, neighbor in self.G.edges():
            flow = self.dissemination_speed * (
                self.G.nodes[neighbor]["water_level"]
                - self.G.nodes[node]["water_level"]
            )
            self.G[node][neighbor]["water_flow"] = abs(flow)
            new_levels[node] += flow
            new_levels[neighbor] -= flow
        nx.set_node_attributes(self.G, new_levels, "water_level")

    def simulate(self, num_steps):
        self.initialize_water()
        self.node_history = [dict(nx.get_node_attributes(self.G, "water_level"))]
        self.edge_history = [dict(nx.get_edge_attributes(self.G, "water_flow"))]
        for _ in range(num_steps):
            self.update_network()
            self.node_history.append(
                dict(nx.get_node_attributes(self.G, "water_level"))
            )
            self.edge_history.append(dict(nx.get_edge_attributes(self.G, "water_flow")))

    def get_node_history(self):
        return self.node_history

    def get_edge_history(self):
        return self.edge_history

    def get_network(self):
        return self.G


def plot_network_states(G, node_history, edge_history, num_plots=5):
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 4))
    steps = np.linspace(0, len(node_history) - 1, num_plots, dtype=int)
    pos = nx.spring_layout(G)
    for i, step in enumerate(steps):
        water_levels = list(node_history[step].values())
        edge_flows = list(edge_history[step].values())
        nx.draw(
            G,
            pos,
            ax=axes[i],
            node_color=water_levels,
            edge_color=edge_flows,
            cmap="Blues",
            edge_cmap=plt.cm.Reds,
            node_size=100,
            width=2,
            vmin=0,
            vmax=max(water_levels),
            edge_vmin=0,
            edge_vmax=max(edge_flows),
        )
        axes[i].set_title(f"Step {step}")
    fig.colorbar(plt.cm.ScalarMappable(cmap="Blues"), ax=axes, label="Node Water Level")
    fig.colorbar(plt.cm.ScalarMappable(cmap="Reds"), ax=axes, label="Edge Water Flow")
    plt.tight_layout()
    plt.show()


def plot_over_time(history, ylabel, title):
    plt.figure(figsize=(12, 6))
    for key in history[0].keys():
        plt.plot(
            range(len(history)),
            [step[key] for step in history],
            label=f"{ylabel.split()[0]} {key}",
        )
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim = WaterDisseminationSimulation("erdos_renyi", 20, 0, 0.1, p=0.1)
    sim.simulate(500)
    plot_network_states(
        sim.get_network(), sim.get_node_history(), sim.get_edge_history()
    )
    plot_over_time(
        sim.get_node_history(), "Water Level", "Water Level of Each Node Over Time"
    )
    plot_over_time(
        sim.get_edge_history(), "Water Flow", "Water Flow on Each Edge Over Time"
    )
