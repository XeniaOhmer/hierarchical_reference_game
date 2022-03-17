
import numpy as np
from matplotlib import pyplot as plt


def plot_heatmap(result_list,
                 mode,
                 plot_dims=(2, 2),
                 figsize=(7, 7),
                 ylims=(0.6, 1.0),
                 titles=('train', 'validation', 'zero shot objects', 'zero shot abstractions'),
                 suptitle=None,
                 suptitle_position=1.03,
                 different_ylims=False,
                 n_runs=5,
                 matrix_indices=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
                 fontsize=18):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
    Allows for plotting multiple matrices according to plot_dims, and allows different modes:
    'max', 'min', mean', 'median', each across runs. """

    plt.figure(figsize=figsize)

    for i in range(np.prod(plot_dims)):

        if different_ylims:
            y_lim = ylims[i]
        else:
            y_lim = ylims

        heatmap = np.empty((3, 3))
        heatmap[:] = np.nan
        results = result_list[i]
        if results.shape[-1] > n_runs:
            results = results[:, :, -1]

        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        if mode == 'mean':
            values = np.nanmean(results, axis=-1)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            heatmap[pos] = values[p]

        im = plt.imshow(heatmap, vmin=y_lim[0], vmax=y_lim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('# values', fontsize=fontsize)
        plt.ylabel('# attributes', fontsize=fontsize)
        plt.xticks(ticks=[0, 1, 2], labels=[4, 8, 16], fontsize=fontsize-1)
        plt.yticks(ticks=[0, 1, 2], labels=[3, 4, 5], fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(y_lim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for k in range(3):
            for l in range(3):
                if not np.isnan(heatmap[l, k]):
                    ax = plt.gca()
                    _ = ax.text(l, k, np.round(heatmap[k, l], 2), ha="center", va="center", color="k",
                                fontsize=fontsize)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, y=suptitle_position)

    plt.tight_layout()


def plot_heatmap_different_vs(result_list,
                              mode,
                              plot_dims=(2, 2),
                              figsize=(7, 9),
                              ylims=(0.6, 1.0),
                              titles=('train', 'validation', 'zero shot objects', 'zero shot abstractions'),
                              suptitle=None,
                              suptitle_position=1.03,
                              n_runs=5,
                              matrix_indices=((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
                              different_ylims=False,
                              fontsize=18,
                              ):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
        Allows for plotting multiple matrices according to plot_dims, and allows different modes:
        'max', 'min', mean', 'median', each across runs.
    """

    plt.figure(figsize=figsize)

    for i in range(np.prod(plot_dims)):

        if different_ylims:
            ylim = ylims[i]
        else:
            ylim = ylims

        heatmap = np.empty((4, 2))
        heatmap[:] = np.nan
        results = result_list[i]
        if results.shape[-1] > n_runs:
            results = results[:, :, -1]

        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        if mode == 'mean':
            values = np.nanmean(results, axis=-1)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            try:
                heatmap[pos] = values[p]
            except:
                continue

        im = plt.imshow(heatmap, vmin=ylim[0], vmax=ylim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('balanced', fontsize=fontsize)
        plt.ylabel('vocab size factor', fontsize=fontsize)
        plt.xticks(ticks=[0, 1], labels=['True', 'False'], fontsize=fontsize-1)
        plt.yticks(ticks=[0, 1, 2, 3], labels=[1, 2, 3, 4], fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.05, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(ylim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for k in range(4):
            for l in range(2):
                if not np.isnan(heatmap[k, l]):
                    ax = plt.gca()
                    _ = ax.text(l, k, np.round(heatmap[k, l], 2), ha="center", va="center", color="k",
                                fontsize=fontsize)
        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, x=0.51, y=suptitle_position)
    plt.tight_layout()


def plot_training_trajectory(results_train,
                             results_val,
                             steps=(1, 5),
                             figsize=(10, 7),
                             ylim=None,
                             plot_indices=(1, 2, 3, 4, 5, 7),
                             plot_shape=(3, 3),
                             n_epochs=300,
                             titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)')):
    """ Plot the training trajectories for training and validation data"""
    plt.figure(figsize=figsize)

    for i, plot_idx in enumerate(plot_indices):
        plt.subplot(plot_shape[0], plot_shape[1], plot_idx)
        plt.plot(range(0, n_epochs, steps[0]), np.transpose(results_train[i]), color='blue')
        plt.plot(range(0, n_epochs, steps[1]), np.transpose(results_val[i]), color='red')
        plt.legend(['train', 'val'])
        plt.title(titles[i], fontsize=13)
        leg = plt.legend(['train', 'val'], fontsize=12)
        leg.legendHandles[0].set_color('blue')
        leg.legendHandles[1].set_color('red')
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        if ylim:
            plt.ylim(ylim)

    plt.suptitle('accuracy', x=0.53, fontsize=15)
    plt.tight_layout()
