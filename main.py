import numpy as np
import pandas as pd

# KNN (K Nearest Neighbors) Alforithm
# pros:
# 1. High accuracy (with balanced dataset)
# 2. Easy to understand
# cons:
# 1. Time complexity O(n)
# 2. Consume huge memory cache
# 3. Can NOT handle with unbalanced dataset

class KNN():
    '''
    KNN algorithm
    '''

    def __init__(self, k):
        '''
        Args:
            k(int): The nearest k instances
        '''
        self.k = k

    def train_data_loader(self, train_path, label_name='Species'):
        '''
        Load training dataset
        Args:
            train_path(string): File path of training dataset
            label_name(string): Label name of the given dataset
        '''
        train_csv = pd.read_csv(
            train_path, header=-1, names=CSV_COLUMN_NAMES).sample(frac=1).reset_index(drop=True)
        # Split the loaded training dataset into features and labels
        train_fs, self.train_ls = train_csv, train_csv.pop(label_name)
        # Normalize features
        self.norm_train_fs = (train_fs - train_fs.min()) / \
            (train_fs.max() - train_fs.min())

        return self.norm_train_fs, self.train_ls

    def test_data_loader(self, test_path, label_name='Species'):
        '''
        Load testing dataset
        Args:
            test_path(string): File path of testing dataset
            label_name(string): Label name of the given name
        '''
        test_csv = pd.read_csv(
            test_path, header=-1, names=CSV_COLUMN_NAMES).sample(frac=1).reset_index(drop=True)
        # Split the loaded testing dataset into features and labels
        test_fs, self.test_ls = test_csv, test_csv.pop(label_name)
        # Normalize features
        self.norm_test_fs = (test_fs - test_fs.min()) / \
            (test_fs.max() - test_fs.min())

        return self.norm_test_fs, self.test_ls

    def pred(self, test_f):
        '''
        Predict the label of each testing
        Args:
            test_f ( < numpy.ndarray > ): Features dataframe of testing dataset
        '''
        feat_dist = []
        # Calculate the feature distances of given data points `test_f`
        # from the testing dataset `test_fs`
        for f in self.norm_train_fs.values:
            feat_dist.append(sum(map(abs, f - test_f)))
        # Binding feature distances with training labels
        _ = pd.DataFrame({"F": feat_dist, "L": self.train_ls})
        # Sorting above dataframe by features distance from low to high
        # Return the first k training labels
        _ = _.sort_values(by='F')['L'][0:self.k].values

        return _


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
TRAIN_PATH, TEST_PATH = 'train_iris.data.csv', 'test_iris.data.csv'
SPECIES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Initialization
accuracy = []
# K: from 1 to len(train_fs)
for k in range(140):
    knn = KNN(k=k + 1)
    # Load data
    train_fs, train_ls = knn.train_data_loader(TRAIN_PATH)
    test_fs, test_ls = knn.test_data_loader(TEST_PATH)

    correct = 0  # Number of the correct predictions
    for i, test_f in enumerate(test_fs.values, 0):
        _ = knn.pred(test_f)
        count = [list(_).count('Iris-setosa'),
                 list(_).count('Iris-versicolor'), list(_).count('Iris-virginica')]
        print('Distribution: {}'.format(count))
        mode = SPECIES[count.index(max(count))]
        if mode == test_ls[i]:
            correct += 1
        print('Predict: {}'.format(mode), 'TEST_LABEL: {}'.format(test_ls[i]),)
    accuracy.append(correct / len(test_fs))

for (i, each_acc) in enumerate(accuracy, 0):
    print('k: {}'.format(i + 1), 'Accuracy: {}'.format(each_acc))
