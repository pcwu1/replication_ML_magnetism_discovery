import matplotlib.pyplot as plt
import pandas as pd

# Read in the predictions
y_test = pd.read_csv("y_test.csv")
y_pred = pd.read_csv("test_predictions.csv")

print(y_test.info())
print(y_pred.info())

# Sort both lists by ascending order by y_test
y_test_sorted = y_test.sort_values(by='target')
y_pred_sorted = y_pred.reindex(y_test_sorted.index)

# Reset the index
y_test_sorted.reset_index(drop=True, inplace=True)
y_pred_sorted.reset_index(drop=True, inplace=True)

# Drop old indices
y_test_sorted.drop(columns=['Unnamed: 0'], inplace=True)
y_pred_sorted.drop(columns=['Unnamed: 0'], inplace=True)

print(y_test_sorted[:10])
print(y_pred_sorted[:10])

# Plot the sorted predictions against the true values
fig = plt.figure(figsize=(12, 6), dpi=600)
ax = fig.add_subplot(111)

ax.axvline(x=25, color = 'green', linestyle = '--')
ax.axvline(x=332, color = 'green', linestyle = '--')

ax.set_title("Original output")
ax.set_xlabel('Sample Number')
ax.set_ylabel('Magnetism')

ax.plot(y_test_sorted,'r', label = "Actual Value",linewidth=0.5)
ax.plot(y_pred_sorted, label = "Predicted Value", linewidth=0.5)
ax.legend()

fig.savefig("../images/output/original_output.svg", dpi = 1200)