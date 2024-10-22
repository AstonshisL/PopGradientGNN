from scipy.spatial.distance import cosine
import numpy as np
from brainspace.gradient import GradientMaps


def zscore_normalize(matrix):
    mean = np.mean(matrix, axis=1, keepdims=True)
    std = np.std(matrix, axis=1, keepdims=True)
    zscore_normalized = (matrix - mean) / std
    return zscore_normalized


def sparse_top_percentage(matrix, percentage=10):
    threshold = np.percentile(matrix, 100 - percentage, axis=1, keepdims=True)
    sparse_matrix = np.where(matrix >= threshold, matrix, 0)
    return sparse_matrix


def cosine_similarity(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            similarity = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix


def gradient_zscore(sequence):
    mean = np.mean(sequence)
    std = np.std(sequence)
    normalized_sequence = (sequence - mean) / std
    return normalized_sequence


def get_gradients(input_matrix):
    # zscore归一化
    normalized_matrix = zscore_normalize(input_matrix)
    # 保留每一行的前10%进行稀疏处理
    sparse_matrix = sparse_top_percentage(normalized_matrix, percentage=10)
    # 对稀疏处理后的矩阵的每一行使用余弦距离计算求相关性
    similarity_matrix = cosine_similarity(sparse_matrix)
    gm = GradientMaps(n_components=4, random_state=0)
    gm.fit(similarity_matrix)
    gradients = np.concatenate(
        (
            gm.gradients_[:, 0],
            #gm.gradients_[:, 1],
            #gm.gradients_[:, 2],
            #gm.gradients_[:, 3],
        ),
        axis=0,
    )

    return gradients
