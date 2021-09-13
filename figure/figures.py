import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_keyword_heatmap(accs_sets):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    accs_fine_grid = np.array(accs_sets)
    nan_mask = np.isnan(accs_sets)
    sns.heatmap(accs_sets, vmin=0, vmax=4, mask=nan_mask, annot=True, fmt='g',
                yticklabels=range(1, 6), xticklabels=range(1, 6), ax=axes[0], cbar=True)

    axes[0].set_ylabel('Tested on Task')
    axes[0].set_xlabel('Naive')
    plt.show()


if __name__ == "__main__":
    accs_native = []
    accs_sets = [[1, np.nan, np.nan, np.nan, np.nan], [1, 1, np.nan, np.nan, np.nan], [2, 2, 2, np.nan, np.nan], [3, 3, 3, 3, np.nan], [4, 4, 4, 4, 4]]
    plot_keyword_heatmap(accs_sets)