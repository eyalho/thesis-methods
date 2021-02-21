from typing import Dict

import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_tsne(x: np.ndarray, y: np.ndarray, class_to_label: Dict[int, str]):
    RS = 20150101
    tsne_projection = TSNE(random_state=RS).fit_transform(x)
    f, ax, sc, txts = plot_2d_projection(tsne_projection, y, class_to_label)
    return tsne_projection, (f, ax, sc, txts)


def plot_2d_projection(proj_2d_x: np.ndarray, y: np.ndarray, class_to_label: Dict[int, str]):
    palette = np.array(sns.color_palette("hls", len(np.unique(y))))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(proj_2d_x[:, 0], proj_2d_x[:, 1], s=40, c=palette[y.astype(np.int)])
    ax.axis('tight')

    # We add the labels for each class.
    txts = []
    for i in range(len(np.unique(y))):
        # Position of each label.
        xtext, ytext = np.median(proj_2d_x[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, class_to_label.get(i, "no_label"), fontsize=18,
                      bbox=dict(facecolor=palette[i], alpha=0.8))
        txts.append(txt)

    return f, ax, sc, txts
