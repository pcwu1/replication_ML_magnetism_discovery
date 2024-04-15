import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imbalance_funct import imbalance_check

# read csv
train = pd.read_csv('../train_test/train.csv', header = 0)
test = pd.read_csv('../train_test/test.csv', header = 0)

stat_results = imbalance_check(train, test, save_name="train_test_imbalance", save=True)
print(stat_results)