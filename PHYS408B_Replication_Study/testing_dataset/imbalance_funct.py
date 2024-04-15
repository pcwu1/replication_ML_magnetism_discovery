import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_stats(dataset):
    dataset_size = dataset.shape[0]
    dataset_mean = np.mean(dataset["M"])
    dataset_median = np.median(dataset["M"])
    dataset_std = np.std(dataset["M"])
    dataset_min = np.min(dataset["M"])
    dataset_max = np.max(dataset["M"])

    summary_df = pd.DataFrame({
        "Size": [dataset_size],
        "Mean": [dataset_mean],
        "Median": [dataset_median],
        "STD": [dataset_std],
        "Min": [dataset_min],
        "Max": [dataset_max]
    })

    return summary_df

def imbalance_check(dataset1, dataset2, save_name="Temp_name", save=False):

    train_sorted = dataset1.sort_values(by="M")
    test_sorted = dataset2.sort_values(by="M")

    train_sorted.reset_index(drop=True, inplace=True)
    test_sorted.reset_index(drop=True, inplace=True)

    # Calculate percentage values for x-axis
    train_percent = np.linspace(0, 100, len(train_sorted))
    test_percent = np.linspace(0, 100, len(test_sorted))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Plot training set on the first subplot
    ax1.plot(train_percent, train_sorted["M"], label=f"Data Set 1 ({dataset1.shape[0]} samples)", color="r")
    ax1.set_title("Checking for imbalanced data")
    ax1.set_ylabel("Magnetism")
    ax1.legend()

    # Plot test set on the second subplot
    ax2.plot(test_percent, test_sorted["M"], label=f"Data Set 2 ({dataset2.shape[0]} samples)")
    ax2.set_ylabel("Magnetism")
    ax2.legend()

    # Plot test set on the second subplot
    ax3.plot(train_percent, train_sorted["M"], label="Data Set 1", color="r")
    ax3.plot(test_percent, test_sorted["M"], label="Data Set 2")
    ax3.set_xlabel("Data Point Position")
    ax3.set_ylabel("Magnetism")
    ax3.legend()

    # Save the graphs
    if save == True:
        # fig.savefig(f"../images/{save_name}.png", dpi=600)
        # fig.savefig(f"../images/{save_name}.eps", format='eps')
        fig.savefig(f"../images/imbalance/{save_name}.svg", format='svg', dpi=1200)

    # Calculate the stats for the datasets
    train_summary = calculate_stats(dataset1)
    test_summary = calculate_stats(dataset2)

    # Concatenate the two dataframes of summaries together and return
    summary_df = pd.concat([train_summary, test_summary], keys=["Data Set 1", "Data Set 2"]).reset_index(level=1, drop=True)

    return summary_df