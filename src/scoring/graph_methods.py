import networkx as nx
import numpy as np
import scipy
import torch
from scipy.linalg import fractional_matrix_power as fmp

EPS = 0.0001


def scale_entropy(entropy, n_classes):
    max_entropy = -np.log(1.0 / n_classes)  # For a discrete distribution with num_classes
    scaled_entropy = entropy / max_entropy
    return scaled_entropy


def vn_entropy(K, normalize=True, scale=True, jitter=0):
    if normalize:
        K = normalize_kernel(K) / K.shape[0]
    result = 0
    eigvs = np.linalg.eig(K + jitter * np.eye(K.shape[0])).eigenvalues.astype(np.float64)
    for e in eigvs:
        if np.abs(e) > 1e-8:
            result -= e * np.log(e)
    if scale:
        result = scale_entropy(result, K.shape[0])
    return np.float64(result)


def get_laplacian(G, norm_lapl):
    if isinstance(G, nx.DiGraph):
        L = nx.directed_laplacian_matrix(G)
    elif norm_lapl:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    return L


def heat_kernel(G: nx.Graph, t: float = 0.4, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    return scipy.linalg.expm(-t * L)


def matern_kernel(G: nx.Graph, kappa: float = 1, nu=1, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    I = np.eye(L.shape[0])
    # return fmp(nu * I + L, -alpha / 2) @ fmp(nu * I + L.T, -alpha / 2)
    return fmp((2 * nu / kappa ** 2) * I + L, -nu)


def normalize_kernel(K):
    diagonal_values = np.sqrt(np.diag(K)) + EPS
    normalized_kernel = K / np.outer(diagonal_values, diagonal_values)
    return normalized_kernel


def build_graph(model, sentences: list[str], threshold=None) -> nx.DiGraph:
    distance_matrix = similarity_matrix_from_sbert(model, sentences)
    graph = graph_from_distance_matrix(distance_matrix, threshold)
    return graph


def similarity_matrix_from_sbert(model, sentences: list[str], threshold=None) -> np.ndarray:
    """
    Build a directed graph from the sentences.
    Each sentence is a node, and edges are created based on the SBERT similarity between sentences.
    """

    # Compute SBERT embeddings for the sentences

    embeddings = model.encode(sentences)

    distances = np.abs(embeddings @ embeddings.T)

    if threshold is not None:
        distances[distances < threshold] = 0
    return distances


def graph_from_distance_matrix(distance_matrix):
    """
    Create a directed graph from a distance matrix.
    """

    return nx.from_numpy_array(distance_matrix, create_using=nx.DiGraph)
