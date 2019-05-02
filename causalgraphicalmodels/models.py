import copy
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from tqdm import tqdm
from .cgm import CausalGraphicalModel


class NodeBaseEstimator(BaseEstimator):
    def sample(self, X):
        raise NotImplementedError


class NoParentClassifier(NodeBaseEstimator, ClassifierMixin):
    """
    Classifier for a node with no parents
    """
    def __init__(self):
        self.data = None

    def fit(self, X, y):
        self.data = y
        return self

    def predict(self, X):
        pass


class NoParentRegressor(NodeBaseEstimator, RegressorMixin):
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


class NodeModel:
    """
    A wrapper that allows the Causal model to fit all nodes with the correct input.
    Input models are assumed to be sklearn compatible.
    """
    def __init__(self, name, model, node_type='continuous'):
        self.name = name
        self.model = model
        self.node_type = node_type
        self.parent_nodes = []

    def __repr__(self):
        return '"{}" ~ {}'.format(self.name, self.model)

    def predict(self, X):
        return self.model.predict(X)


class CausalModel:
    """
    This implements _all_ models required to specify the _entire_ graph.
    """
    def __init__(self, data: pd.DataFrame, node_models: list, edges: list, latent_edges=None, set_nodes=None):
        self.data = data.copy()
        self.models = {node.name: node for node in node_models}
        self.graphical_model: CausalGraphicalModel = CausalGraphicalModel(
            [node.name for node in node_models],
            edges,
            latent_edges,
            set_nodes
        )

    def solve(self):
        """
        Solves all Models in the graphs

        :return:
        """
        solve_order, node_parents = [], {}
        for node in self.graphical_model.dag.nodes:
            ancestors = list(nx.ancestors(self.graphical_model.dag, node))
            predecessors = list(self.graphical_model.dag.predecessors(node))

            solve_order.append((node, len(ancestors)))
            node_parents[node] = predecessors

        # Compute solve order
        solve_order.sort(key=lambda x: x[1])

        # Fit all models in the network
        solve_progressor = tqdm(solve_order)
        for node, _ in solve_progressor:
            solve_progressor.set_description('Solving {}'.format(node))

            y = self.data[node]
            x = self.data[node_parents[node]]

            # TODO (5/2/2019) implement automatic imputation

            if x.shape[1] == 0:
                # Has no inputs
                # TODO 5/1/2019 Allow classifiers
                self.models[node].model = NoParentRegressor().fit(x, y)
            else:
                self.models[node].model.fit(x, y)
                self.models[node].parent_nodes = node_parents[node]

            solve_progressor.set_description('Done')

    def do(self, x, equal_to):
        """
        Make new causal graph with the do operator on x equal to x_do

        do(X=x)

        This also simulates the `do` operation and therefore recomputes data for using knowledge about the graph

        :param x: Which node to perform the do operator on
        :param equal_to: What value to make `x` do
        :return:
        """
        new = copy.deepcopy(self)
        new.data[x] = equal_to
        new.graphical_model = new.graphical_model.do(x)

        # TODO (5/2/2019) to simulate the update correct you need to estimate values both before and after and move the
        #  obvserved values by the difference.

        return new

    def estimate(self, x, df=None):
        """
        Estimate the values of the node x given the current causal graph and initial data (unless a new df is given)

        :param x:
        :param  df:
        :return:
        """
        # TODO (5/2/2019) this gives you wrong answers when the do operator was performed
        #  because it does not propagate effects through the graph.
        df = self.data if df is None else df
        return self.models[x].predict(df[self.models[x].parent_nodes])

    def draw(self):
        return self.graphical_model.draw()
