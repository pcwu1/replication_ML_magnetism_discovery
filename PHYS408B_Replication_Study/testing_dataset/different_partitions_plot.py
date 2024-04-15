import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv("different_partitions_data.csv")

fig, axs = plt.subplots(3, 1, figsize=(10, 8))

num_samples = data_df["num_samples"]
mse = data_df["mse"]
r2 = data_df["r2"]
mae = data_df["mae"]

# MSE
axs[0].plot(num_samples, mse, label="MSE", color="blue")
axs[0].scatter(num_samples, mse, color="blue", marker="o")
axs[0].set_title("Mean Squared Error (MSE)")
axs[0].set_xlabel("Number of Samples")
axs[0].set_ylabel("MSE")
axs[0].set_xticks(num_samples)
axs[0].legend()

# R2
axs[1].plot(num_samples, r2, label="R2", color="red")
axs[1].scatter(num_samples, r2, color="red", marker="o")
axs[1].set_title("R-squared (R2)")
axs[1].set_xlabel("Number of Samples")
axs[1].set_ylabel("R2")
axs[1].set_xticks(num_samples)
axs[1].legend()

# MAE
axs[2].plot(num_samples, mae, label="MAE", color="green")
axs[2].scatter(num_samples, mae, color="green", marker="o")
axs[2].set_title("Mean Absolute Error (MAE)")
axs[2].set_xlabel("Number of Samples")
axs[2].set_ylabel("MAE")
axs[2].set_xticks(num_samples)
axs[2].legend()

plt.tight_layout()
plt.show()

fig.savefig("../images/metric_comparison_for_partitions.svg", dpi=1200)
