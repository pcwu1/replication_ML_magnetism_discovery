import pandas as pd
from distribution_funct import distribution_check

# read csv
train = pd.read_csv('../train_test/train.csv', header = 0)
test = pd.read_csv('../train_test/test.csv', header = 0)

# check the distribution of the train and test data
stat_results = distribution_check(train, test, save_name="train_test_distribution", save=True)
print(stat_results)