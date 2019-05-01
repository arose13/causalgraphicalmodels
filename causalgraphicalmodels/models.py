import numpy as np
import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
from .cgm import CausalGraphicalModel


class NoParentRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor for a node with no parents
    """
    def __init__(self):
        self.data = None

    def fit(self, X, y):
        self.data = y
        return self

    def predict(self, X):
        return np.median(self.data) * np.ones_like(self.data)

    def sample(self, X):
        raise NotImplementedError


class NodeModel:
    """
    A wrapper that allows the Causal model to fit all nodes with the correct input.
    Input models are assumed to be sklearn compatible.
    """
    def __init__(self, name, model, node_type='continuous'):
        self.name = name
        self.model = model
        self.node_type = node_type


class CausalModel:
    """
    This implements _all_ models required to specify the _entire_ graph.
    """
    def __init__(self, data: pd.DataFrame, node_models: list, edges: list, latent_edges=None, set_nodes=None):
        self.data = data
        self.models = {node.name: node for node in node_models}
        self.graphical_model = CausalGraphicalModel(
            [node.name for node in node_models],
            edges,
            latent_edges,
            set_nodes
        )

    def fit(self):
        """
        Solves all Models in the graphs

        :return:
        """
        solve_order, node_parents = [], {}
        for node in self.graphical_model.dag.nodes:
            ancestors = list(nx.ancestors(self.graphical_model.dag, node))

            solve_order.append((node, len(ancestors)))
            node_parents[node] = ancestors

        # Compute solve order
        solve_order.sort(key=lambda x: x[1])

        # Fit all models in the network
        solve_progressor = tqdm(solve_order)
        for node, _ in solve_progressor:
            solve_progressor.set_description('Solving {}'.format(node))

            y = self.data[node]
            x = self.data[node_parents[node]]

            if x.shape[1] == 0:
                # Has no inputs
                # TODO 5/1/2019 Allow classifiers
                self.models[node].model = NoParentRegressor().fit(x, y)
            else:
                self.models[node].model.fit(x, y)

            solve_progressor.set_description('Done')