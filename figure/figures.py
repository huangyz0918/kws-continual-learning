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

    line_width = 2
    ax.plot(index, si_acc_list, color=COLOR_LIST[3], ms=10, linewidth=line_width, label='SI')
    ax.plot(index, gem_acc_list, color=COLOR_LIST[5], ms=10, linewidth=line_width, label='GEM')
    ax.plot(index, tcpnn_acc_list, color=COLOR_LIST[2], ms=10, linewidth=line_width, label='TC-PNN')
    ax.plot(index, single_acc_list, color=COLOR_LIST[0], ms=10, linewidth=line_width, label='Stand-alone')
    ax.plot(index, nr_acc_list, color=COLOR_LIST[4], ms=10, linewidth=line_width, label='Native Rehearsal')
    ax.plot(index, tune_acc_list, color=COLOR_LIST[6], ms=10, linewidth=line_width, label='Fine-tune')

    plt.margins(x=0.08)

    ax.set_xlabel('Task Number', fontsize=16)
    ax.set_ylabel('ACC (%)', fontsize=16)
    ax.set_ylim([-19, 110])

    ax.tick_params(axis='x', rotation=0)
    ax.set_xticks(index)
    ax.set_xticklabels(label, fontsize=13)
    ax.legend(loc='lower left', fontsize=13, ncol=2)
    plt.savefig(f'./task_avg_acc.pdf', format='pdf', bbox_inches='tight')


def toK(input_list):
    return [x / 1000 for x in input_list]


def plot_param():
    """
    Plot the Extra Parameters after learning each new task.
    """
    sl_list = [0, 67, 67 * 2, 67 * 4, 67 * 8, 67 * 16, 67 * 32, 67 * 64, 67 * 128, 67 * 256]
    tcpnn_fix_list = [0, 34, 69, 137, 276, 558, 1141, 2380, 5152, 11872]
    tcpnn_list = [0, 3, 6, 13, 27, 56, 119, 265, 634, 1684]
    labels = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))
    index = np.arange(len(labels))

    line_width = 2
    ax.plot(index, toK(sl_list), color=COLOR_LIST[3], ms=10, linewidth=line_width, label='Stand-alone')
    ax.plot(index, toK(tcpnn_fix_list), color=COLOR_LIST[4], ms=10, linewidth=line_width, label='TC-PNN (fix)')
    ax.plot(index, toK(tcpnn_list), color=COLOR_LIST[2], ms=10, linewidth=line_width, label='TC-PNN')
    plt.text(x=0, y=32 * 128 / 1000 + 0.5, s="Buffer size of GEM-128", fontsize=17, color="#A52A2A")
    plt.axhline(y=32 * 128 / 1000, color=COLOR_LIST[0], linewidth=2, xmin=0)
    # plt.text(x=0, y=32*512/1000 + 0.1, s="GEM-512 buffer size", fontsize=13, color="#A52A2A")
    # plt.axhline(y=32*512/1000, color=COLOR_LIST[0], linewidth=2, xmin=0)

    plt.margins(x=0.08)

    ax.set_xlabel('Task Number', fontsize=17)
    ax.set_ylabel('Extra Param (M)', fontsize=17)
    # ax.set_ylim([-19, 110])

    ax.tick_params(axis='x', rotation=0)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, fontsize=13)
    ax.legend(loc='upper left', fontsize=17)
    plt.savefig(f'./task_param.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    set_style()

    plot_param()
    # accs_native = []
    # accs_sets = [[1, np.nan, np.nan, np.nan, np.nan], [1, 1, np.nan, np.nan, np.nan], [2, 2, 2, np.nan, np.nan],
    #              [3, 3, 3, 3, np.nan], [4, 4, 4, 4, 4]]
    # plot_keyword_heatmap(accs_sets)
    # plot_avg_acc()
