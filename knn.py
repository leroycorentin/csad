import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Prepare Dataset
# load data
train = pd.read_csv("digit-recognizer/train.csv", dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values

# train test split. Size of train data is 80% and size of test data is 20%.
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42)

nb_neighbors = 7
KNN = KNeighborsClassifier(nb_neighbors)
KNN.fit(features_train, targets_train)
print(KNN.score(features_test, targets_test))
