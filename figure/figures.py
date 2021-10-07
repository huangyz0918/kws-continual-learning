import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from figure.util import set_style, set_size, WIDTH, COLOR_LIST


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
    gem_acc_list = [96.2, 79.1, 74.2, 71.3, 62.4, 52.9]
    # nr_acc_list = [96.2, 91.7, 92.5, 89.4, 90.1, 88.9] # 100%
    nr_05_acc_list = [96.2, 81.2, 80.4, 77.2, 79.3, 78.9]
    nr_075_acc_list = [96.2, 84.2, 85.5, 80.0, 85.1, 81.4]
    ewc_acc_list = [96.2, 49.6, 32.8, 27.5, 20, 17.1]
    tune_acc_list = [96.2, 51.2, 30.0, 24.1, 19.9, 18.4]
    tcpnn_acc_list = [96.2, 86.1, 91.8, 87.6, 94.2, 86.5]
    single_acc_list = [96.2, 95.0, 98.2, 95.4, 96.6, 97.6]

    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))
    label = range(len(si_acc_list))
    index = np.arange(len(label))

    line_width = 2.5
    ax.plot(index, si_acc_list, color=COLOR_LIST[8], ms=10, linewidth=line_width, label='SI')
    ax.plot(index, ewc_acc_list, color=COLOR_LIST[5], ms=10, linewidth=line_width, label='EWC')
    ax.plot(index, nr_05_acc_list, color=COLOR_LIST[3], ms=10, linewidth=line_width, label='NR-0.5')
    ax.plot(index, nr_075_acc_list, color=COLOR_LIST[6], ms=10, linewidth=line_width, label='NR-0.75')
    ax.plot(index, gem_acc_list, color=COLOR_LIST[4], ms=10, linewidth=line_width, label='GEM-128')
    ax.plot(index, tcpnn_acc_list, color=COLOR_LIST[2], ms=10, linewidth=line_width, label='PCL-KWS')
    ax.plot(index, single_acc_list, color=COLOR_LIST[0], ms=10, linestyle='dashed', linewidth=line_width,
            label='Stand-alone')
    ax.plot(index, tune_acc_list, color=COLOR_LIST[1], ms=10, linestyle='dashed', linewidth=line_width,
            label='Fine-tune')

    plt.margins(x=0.08)

    ax.set_xlabel('Task Number', fontsize=16)
    ax.set_ylabel('ACC (%)', fontsize=16)
    ax.set_ylim([-39, 110])

    ax.tick_params(axis='x', rotation=0)
    ax.set_xticks(index)
    ax.set_xticklabels(label, fontsize=14)

    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[1].label1.set_visible(False)

    ax.set_yticklabels([-20, 0, 0, 20, 40, 60, 80, 100])

    ax.legend(loc='lower left', fontsize=13, ncol=3)
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

    line_width = 4
    ax.plot(index, toK(sl_list), color=COLOR_LIST[3], ms=10, linestyle='dashed', linewidth=line_width,
            label='Stand-alone')
    ax.plot(index, toK(tcpnn_fix_list), color=COLOR_LIST[4], ms=10, linewidth=line_width, label='PCL-KWS (fix)')
    ax.plot(index, toK(tcpnn_list), color=COLOR_LIST[2], ms=10, linewidth=line_width, label='PCL-KWS')
    plt.text(x=-0.4, y=32 * 128 / 1000 + 0.5, s="Buffer size of GEM-128", fontsize=22, color="#A52A2A")
    plt.axhline(y=32 * 128 / 1000, color=COLOR_LIST[0], linestyle='dashed', linewidth=line_width, xmin=0)
    # plt.text(x=0, y=32*512/1000 + 0.1, s="GEM-512 buffer size", fontsize=13, color="#A52A2A")
    # plt.axhline(y=32*512/1000, color=COLOR_LIST[0], linewidth=2, xmin=0)

    plt.margins(x=0.08)

    ax.set_xlabel('Task Number', fontsize=22)
    ax.set_ylabel('Sub-network Param (M)', fontsize=22)
    # ax.set_ylim([-19, 110])

    ax.tick_params(axis='x', rotation=0)
    ax.set_xticks(index)
    ax.set_xticklabels(labels)
    ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    # ax.tick_params(axis='x', which='major', labelsize=18)
    # ax.tick_params(axis='y', which='major', labelsize=18)
    ax.legend(loc='upper left', fontsize=22)
    plt.savefig(f'./task_param.pdf', format='pdf', bbox_inches='tight')


def plot_param_acc():
    """
    Plot the Extra Parameters and Corresponding ACC of TC-PNN.
    """
    parm_list = [0.9, 3.3, 10, 20, 30, 50, 70]
    acc_3_list = [72.3, 87.4, 95.7, 96.6, 96.9, 97.0, 97.1]
    acc_15_list = [53.8, 71.6, 80.0, 83.1, 84.9, 85.1, 86.4]

    fig, ax = plt.subplots(1, 1, figsize=set_size(WIDTH))

    line_width = 4
    ax.plot(parm_list, acc_3_list, color=COLOR_LIST[3], ms=10, linewidth=line_width, label='3-keyword spotting')
    ax.plot(parm_list, acc_15_list, color=COLOR_LIST[5], ms=10, linewidth=line_width, label='15-keyword spotting')

    plt.margins(x=0.08)

    ax.set_xlabel('Extra Parameter (K)', fontsize=22)
    ax.set_ylabel('ACC (%)', fontsize=22)
    ax.set_ylim([50, 100])

    ax.tick_params(axis='x', rotation=0)
    ax.xaxis.set_ticks(parm_list)
    ax.set_xticklabels(parm_list)
    ax.set_yticklabels([50, 60, 70, 80, 90, 100])
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[1].label1.set_visible(False)
    # ax.tick_params(axis='x', which='major', labelsize=18)
    # ax.tick_params(axis='y', which='major', labelsize=18)
    ax.legend(loc='lower right', fontsize=22)
    plt.savefig(f'./task_param_acc.pdf', format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    set_style()

    # plot_param()
    plot_param_acc()
    # accs_native = []
    # accs_sets = [[1, np.nan, np.nan, np.nan, np.nan], [1, 1, np.nan, np.nan, np.nan], [2, 2, 2, np.nan, np.nan],
    #              [3, 3, 3, 3, np.nan], [4, 4, 4, 4, 4]]
    # plot_keyword_heatmap(accs_sets)
    # plot_avg_acc()
