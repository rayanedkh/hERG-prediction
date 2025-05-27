import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity, TanimotoSimilarity, ExplicitBitVect


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from collections import defaultdict, Counter


def reduce_fingerprints_by_modulo(fp_array, target_dim=256, binarize=True):
    """
    Réduit un vecteur binaire de taille N (ex: 2048) à `target_dim` (ex: 256)
    par bucketing modulaire.

    Parameters:
        fp_array (np.ndarray): shape (n_samples, 2048), dtype=uint8
        target_dim (int): taille réduite souhaitée
        binarize (bool): si True, convertit en vecteur binaire (0/1)
    
    Returns:
        np.ndarray: shape (n_samples, target_dim)
    """
    assert fp_array.shape[1] % target_dim == 0, "La taille originale doit être divisible par target_dim"
    step = fp_array.shape[1] // target_dim
    reduced = fp_array.reshape(fp_array.shape[0], target_dim, step).sum(axis=2)
    if binarize:
        return (reduced > 0).astype('uint8')
    return reduced


def numpy_to_bitvect(vector):
    bitvect = ExplicitBitVect(2048)
    for idx, val in enumerate(vector):
        if val:
            bitvect.SetBit(idx)
    return bitvect


# Fonction pour calculer la matrice de similarité de Tanimoto
def calculate_tanimoto_matrix(fingerprints):
    """
    Calcule la matrice de similarité de Tanimoto pour un ensemble de fingerprints.
    Parameters:
        fingerprints (list of ExplicitBitVect): Liste de fingerprints RDKit.
        Returns:
        np.ndarray: Matrice de similarité de Tanimoto.
    """
    n = len(fingerprints)
    tanimoto_matrix = np.zeros((n, n))

    for i in tqdm(range(n)):
        for j in range(i):
            
            tanimoto_matrix[i, j] = tanimoto_matrix[j, i] = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            
            
    return tanimoto_matrix



def calculate_distance_matrix(efcf_matrix, fcfc_matrix):
    mean_sim_matrix = np.mean([efcf_matrix, fcfc_matrix], axis=0)
    dist_matrix = 1 - mean_sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix





def cluster_hierarchical(distance_matrix, target_cluster_size):
    """
    Effectue un clustering hiérarchique sur une matrice de distance.
    Parameters:
        distance_matrix (np.ndarray): Matrice de distance (1 - matrice de similarité).
        target_cluster_size (int): Taille souhaitée pour les clusters.
    Returns:
        np.ndarray: Labels de cluster pour chaque molécule.
    """
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='complete')  # ou 'complete'

    # Nombre de clusters ≈ N / taille souhaitée
    n_clusters = max(2, int(distance_matrix.shape[0] / target_cluster_size))
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return labels  # cluster_id pour chaque molécule



def cluster_hierarchical_min_size(distance_matrix, target_cluster_size=70, min_size=20):
    """
    Clustering hiérarchique avec contrainte de taille minimale des clusters.
    """
    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method='complete')

    n_clusters = max(2, int(distance_matrix.shape[0] / target_cluster_size))
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    label_counts = Counter(labels)
    small_clusters = [label for label, count in label_counts.items() if count < min_size]

    if not small_clusters:
        return labels  # rien à faire

    print(f"🔸 Fusion de {len(small_clusters)} petits clusters (<{min_size})...")


    # Créer dictionnaire : cluster_id → liste d'indices
    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[label].append(idx)

    # Rechercher les plus proches gros clusters
    for small in small_clusters:
        small_idxs = cluster_to_indices[small]
        small_centroid = distance_matrix[small_idxs].mean(axis=0)

        # Chercher cluster valide le plus proche
        best_label = None
        best_dist = float('inf')

        for other_label, other_idxs in cluster_to_indices.items():
            if other_label == small or label_counts[other_label] < min_size:
                continue
            other_centroid = distance_matrix[other_idxs].mean(axis=0)
            dist = np.linalg.norm(small_centroid - other_centroid)
            if dist < best_dist:
                best_dist = dist
                best_label = other_label

        # Fusion : réaffecter les indices du petit cluster
        for idx in small_idxs:
            labels[idx] = best_label

    return labels






def visualize_clusters(distance_matrix, clusters, method):
    """
    Visualise les clusters en utilisant t-SNE ou PCA.
    Parameters:
        animoto_matrix (np.ndarray): Matrice de similarité de Tanimoto.
        clusters (list): Liste des labels de clusters.
        method (str): Méthode de réduction de dimensionnalité ('t-sne' ou 'pca').
    """
    if method not in ['t-sne', 'pca']:
        raise ValueError("La méthode doit être 't-sne' ou 'pca'.")

    # Réduction de dimensionnalité avec t-SNE
    if method == 't-sne':
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(distance_matrix)
    elif method == 'pca':
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(distance_matrix)

    # Visualisation des clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='rainbow', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Visualisation des Clusters avec {method}')
    plt.xlabel('Composante 1')
    plt.ylabel('Composante 2')
    plt.show()









def split_by_series(smiles_list, labels, cluster_labels, test_size=0.2, random_state=42):
    """
    Divise les données en train/val en respectant les séries chimiques (groupes).
    
    Parameters:
        smiles_list (list): Liste des SMILES.
        labels (array-like): Classes binaires (0/1).
        cluster_labels (array-like): Labels de cluster (ex: série chimique).
        test_size (float): Ratio du jeu de validation.
    
    Returns:
        dict: {'X_train': [...], 'X_val': [...], 'y_train': [...], 'y_val': [...], 'groups': [...]}
    """
    smiles_array = np.array(smiles_list)
    labels_array = np.array(labels)
    groups = np.array(cluster_labels)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(smiles_array, labels_array, groups))

    return {
        'smiles_train': smiles_array[train_idx],
        'y_train': labels_array[train_idx],
        'smiles_test': smiles_array[val_idx],
        'y_test': labels_array[val_idx],
        'groups': groups
    }
