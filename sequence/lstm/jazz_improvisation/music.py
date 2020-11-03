"""Modified from the original music_utils.py file from the Deep Learning Specialization assignment"""
import numpy as np


def data_processing(corpus, values_indices, m = 60, Tx = 30):
    # Cut the corpus into semi-redundant sequences of Tx values
    Tx = Tx
    N_values = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=np.float)
    Y = np.zeros((m, Tx, N_values), dtype=np.float)
    for i in range(m):
        random_idx = np.random.choice(len(corpus) - Tx)
        corp_data = corpus[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1

    # Important: the y vector must not be one-hot encoded to work with PyTorch's loss functions!
    Y = np.argmax(Y, axis=-1)

    return np.asarray(X), np.asarray(Y), N_values
