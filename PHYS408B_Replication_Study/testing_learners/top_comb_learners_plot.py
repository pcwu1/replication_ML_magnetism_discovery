import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

scores = pd.read_csv("combination_scores.csv")
print(scores)

one_model = scores[:6]
two_model = scores[6:21]
three_model = scores[21:41]
four_model = scores[41:56]
five_model = scores[56:62]
six_model = scores[62:63]

scores_lists = [one_model, two_model, three_model, four_model, five_model, six_model]
# for item in scores_lists:
#     print(item)

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection='3d')

colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']

for i, num_model in enumerate(scores_lists):
    x = num_model["MSE"]
    y = num_model["MAE"]
    z = num_model["R2"]
    ax.scatter(x, y, z, color=colors[i], label=f"{i + 1} model(s)", alpha=1)

ax.set_title("3D graph of combinations of models")
ax.legend()
ax.set_xlabel("MSE")
ax.set_ylabel("MAE")
ax.set_zlabel("R2")
plt.show()
fig.savefig("../images/3D_comb_learners.svg", format='svg', dpi=1200)
##########################################

top_scores = (x + y).nsmallest(5).index

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot()
for i, num_model in enumerate(scores_lists):
    x = num_model["MSE"]
    y = num_model["MAE"]
    ax.scatter(x, y, color=colors[i], label=f"{i + 1} model(s)", alpha=1)

ax.set_title("2D graph of combinations of models")
ax.set_xlabel("MSE")
ax.set_ylabel("MAE")

ax.legend()
plt.show()
fig.savefig("../images/2D_comb_learners.svg", format='svg', dpi=1200)

