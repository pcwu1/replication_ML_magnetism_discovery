import pandas as pd
import matplotlib.pyplot as plt

# Read in the metric scores from all the combination of base learners
# For each base learner combination plot its position in 3D and 2D space where the axis are the metrics
# 3D in MSE, R2, MAE
# 2D in MSE and MAE

scores = pd.read_csv("pred_combination_scores.csv")
print(scores)

# Separate the models by the amount
two_model = scores[:15]
three_model = scores[15:35]
four_model = scores[35:50]
five_model = scores[50:56]
six_model = scores[56:57]

scores_lists = [two_model, three_model, four_model, five_model, six_model]
# for item in scores_lists:
#     print(item)
#     print("----")

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection='3d')

colors = ['red', 'blue', 'green', 'orange', 'black']

# Graph the 3D scatter plot
for i, num_model in enumerate(scores_lists):
    x = num_model["MSE"]
    y = num_model["MAE"]
    z = num_model["R2"]
    ax.scatter(x, y, z, color=colors[i], label=f"{i + 2} model(s)", alpha=1)

ax.set_title("3D graph of combinations of models")
ax.legend()
ax.set_xlabel("MSE")
ax.set_ylabel("MAE")
ax.set_zlabel("R2")
plt.show()
fig.savefig("../images/3D_comb_learners.svg", format='svg', dpi=1200)

##########################################

# Graph the 2D scatter plot
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot()
for i, num_model in enumerate(scores_lists):
    x = num_model["MSE"]
    y = num_model["MAE"]
    ax.scatter(x, y, color=colors[i], label=f"{i + 2} model(s)", alpha=1)

ax.set_title("2D graph of combinations of models")
ax.set_xlabel("MSE")
ax.set_ylabel("MAE")

ax.legend()
plt.show()
fig.savefig("../images/2D_comb_learners.svg", format='svg', dpi=1200)

