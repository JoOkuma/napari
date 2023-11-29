from napari.layers import Graph
import numpy as np
import pandas as pd
from napari_graph import UndirectedGraph
from random import randrange


def build_undirected_graph(n_nodes: int, n_neighbors: int, cols: list = ["t", "z", "y", "x"]) -> UndirectedGraph:
    neighbors = np.random.randint(n_nodes, size=(n_nodes * n_neighbors))

    edges = np.stack([np.repeat(np.arange(n_nodes), n_neighbors), neighbors], axis=1)

    nodes_df = pd.DataFrame(
        400 * np.random.uniform(size=(n_nodes, len(cols))),
        columns=cols,
    )
    graph = UndirectedGraph(edges=edges, coords=nodes_df)

    return graph

# undirected

def test_small_2d_graph():
    n_nodes = randrange(10, 21)
    n_neighbors = randrange(3, 6)
    graph = build_undirected_graph(n_nodes, n_neighbors,  ["y", "x"])
    layer = Graph(graph, out_of_slice_display=True)
    assert layer.data.n_edges == n_nodes * n_neighbors
    assert layer.data.n_nodes == n_nodes


def test_large_2d_graph():
    n_nodes = randrange(1000, 2000)
    n_neighbors = randrange(20, 25)
    graph = build_undirected_graph(n_nodes, n_neighbors,  ["y", "x"])
    layer = Graph(graph, out_of_slice_display=True)
    assert layer.data.n_edges == n_nodes * n_neighbors
    assert layer.data.n_nodes == n_nodes


def test_small_3d_graph():
    n_nodes = randrange(10, 21)
    n_neighbors = randrange(3, 6)
    graph = build_undirected_graph(n_nodes, n_neighbors,  ["z", "y", "x"])
    layer = Graph(graph, out_of_slice_display=True)
    assert layer.data.n_edges == n_nodes * n_neighbors
    assert layer.data.n_nodes == n_nodes



def test_large_3d_graph():
    n_nodes = randrange(1000, 2000)
    n_neighbors = randrange(20, 25)
    graph = build_undirected_graph(n_nodes, n_neighbors,  ["z", "y", "x"])
    layer = Graph(graph, out_of_slice_display=True)
    assert layer.data.n_edges == n_nodes * n_neighbors
    assert layer.data.n_nodes == n_nodes



def test_small_4d_graph():
    n_nodes = randrange(10, 21)
    n_neighbors = randrange(3, 6)
    graph = build_undirected_graph(n_nodes, n_neighbors)
    layer = Graph(graph, out_of_slice_display=True)
    assert layer.data.n_edges == n_nodes * n_neighbors
    assert layer.data.n_nodes == n_nodes


def test_large_4d_graph():
    n_nodes = randrange(1000, 2000)
    n_neighbors = randrange(20, 25)
    graph = build_undirected_graph(n_nodes, n_neighbors)
    layer = Graph(graph, out_of_slice_display=True)
    assert layer.data.n_edges == n_nodes * n_neighbors
    assert layer.data.n_nodes == n_nodes
