import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from figure.util import set_style, set_size, WIDTH, COLOR_LIST, MARKER_LIST


def plot_keyword_heatmap(accs_sets):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    accs_fine_grid = np.array(accs_sets)
    nan_mask = np.isnan(accs_sets)
    sns.heatmap(accs_sets, vmin=0, vmax=4, mask=nan_mask, annot=True, fmt='g',
                yticklabels=range(1, 6), xticklabels=range(1, 6), ax=axes[0], cbar=True)

    axes[0].set_ylabel('Tested on Task')
    axes[0].set_xlabel('Naive')
    plt.show()


def plot_avg_acc():
    """
    Plot the average accuracy after learning each new task.
    """
    si_acc_list = [96.2, 87.7, 54.1, 43.8, 29.9, 25.6]
    gem_acc_list = [96.2, 74.4, 73.0, 78.3, 62.9, 58.5]
    nr_acc_list = [96.2, 91.7, 92.5, 89.4, 90.1, 88.9]
    tune_acc_list = [96.2, 49.6, 32.8, 27.5, 20, 17.1]
    tcpnn_acc_list = [96.2, 86.0, 92.8, 84.6, 94.5, 83.1]
    single_acc_list = [96.2, 95.0, 98.2, 95.4, 96.6, 97.6]

    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))
    label = range(len(si_acc_list))
    index = np.arange(len(label))

    line_width = 1.5
    ax.plot(index, si_acc_list, color=COLOR_LIST[3], ms=10, linewidth=line_width, label='SI')
    ax.plot(index, gem_acc_list, color=COLOR_LIST[5], ms=10, linewidth=line_width, label='GEM')
    ax.plot(index, tcpnn_acc_list, color=COLOR_LIST[2], ms=10, linewidth=line_width, label='TC-PNN')
    ax.plot(index, single_acc_list, color=COLOR_LIST[0], ms=10, linewidth=line_width, label='Stand-alone')
    ax.plot(index, nr_acc_list, color=COLOR_LIST[4], ms=10, linewidth=line_width, label='Native Rehearsal')
    ax.plot(index, tune_acc_list, color=COLOR_LIST[6], ms=10, linewidth=line_width, label='Fine-tune')

    plt.margins(x=0.08)

    ax.set_xlabel('Task Number', fontsize=13)
    ax.set_ylabel('Avg. Accuracy (%)', fontsize=13)
    ax.set_ylim([-19, 110])

    ax.tick_params(axis='x', rotation=0)
    ax.set_xticks(index)
    ax.set_xticklabels(label, fontsize=13)
    ax.legend(loc='lower left', fontsize=13, ncol=2)
    plt.savefig(f'./task_avg_acc.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    set_style()
    # accs_native = []
    # accs_sets = [[1, np.nan, np.nan, np.nan, np.nan], [1, 1, np.nan, np.nan, np.nan], [2, 2, 2, np.nan, np.nan],
    #              [3, 3, 3, 3, np.nan], [4, 4, 4, 4, 4]]
    # plot_keyword_heatmap(accs_sets)

    plot_avg_acc()
