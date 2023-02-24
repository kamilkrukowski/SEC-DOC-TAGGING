
import random

import pickle

import torch

with open('output.pkl', 'rb') as handle:
    data = pickle.load(handle)
# data is the  inputs list generate from preprocess.py
y = [i['y'] for i in data][:30]
X = [i for i in data][:30]


def split_data(
        X, y, train_ratio=0.6,
        test_ratio=0.2, val_ratio=0.2,
        min_samples_per_class=1, random_state=None):
    """
        Split the data into train, test, and validation sets
        while ensuring a minimum number of samples per prediction class
        in each split.
        Parameters
        ---------
        X: ndarray
            Input features of shape (n_samples, n_features)
        y: ndarray
            Target labels of shape (n_samples,)
        train_ratio: float, default=0.8
            Ratio of training set size to total dataset size (between 0 and 1)
        test_ratio: float, default=0.2
            Ratio of test set size to total dataset size (between 0 and 1)
        val_ratio: float, default=0.2
            Ratio of validation set size to total dataset size
            (between 0 and 1)
        min_samples_per_class: int, default=1
            Minimum number of samples per prediction class in each split
        random_state: int or None, default=None
                Random seed for reproducibility.

        Returns
        --------
        DataFrame
            Tuple of (X_train, y_train, X_test, y_test, X_val, y_val)

        Notes
        ------
        ratio should add up to 1
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, \
        'Train, validation, and test proportions must add up to 1.0'
    # Calculate number of samples in each class
    num_classes = len(set(y[0]))
    class_counts = torch.sum(torch.stack(y), dim=0)

    # Calculate minimum number of samples required in each class for each split
    train_min_samples = [max(min_samples_per_class, int(
        count*train_ratio)) for count in class_counts]
    test_min_samples = [max(min_samples_per_class, int(
        count*test_ratio)) for count in class_counts]
    val_min_samples = [max(min_samples_per_class, int(
        count*val_ratio)) for count in class_counts]

    # Create empty lists for train, test and validation sets
    train_set = [[] for i in range(num_classes)]
    test_set = [[] for i in range(num_classes)]
    val_set = [[] for i in range(num_classes)]

    # Split samples for each class
    for i in range(num_classes):
        # generate sample for class with index i
        class_samples = [j for j, x in enumerate(
            [val[i] for val in y]) if x == 1]

        # Check if there are enough sample
        min_samples_per_class = train_min_samples[i] + \
            test_min_samples[i] + val_min_samples[i]
        if len(class_samples) < min_samples_per_class:
            print(
                f'Class {i}: has less than {min_samples_per_class} samples')
            continue

        # Shuffle samples
        random.seed(random_state)
        random.shuffle(class_samples)

        # Divide samples into train, test and validation sets
        train_upper = train_min_samples[i]+test_min_samples[i]
        train_samples = class_samples[:train_min_samples[i]]
        test_samples = class_samples[train_min_samples[i]:train_upper]
        val_samples = class_samples[train_upper:min_samples_per_class]

        # Add samples to respective sets
        for j in train_samples:
            train_set[i].append(j)
        for j in test_samples:
            test_set[i].append(j)
        for j in val_samples:
            val_set[i].append(j)

    # Concatenate sets for each class and return final split
    train_idx = [j for indices in train_set for j in indices]
    test_idx = [j for indices in test_set for j in indices]
    val_idx = [j for indices in val_set for j in indices]

    # Shuffle indices
    random.shuffle(train_idx)
    random.shuffle(test_idx)
    random.shuffle(val_idx)

    # Use the indices to split the data
    X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
    X_val, y_val = [X[i] for i in val_idx], [y[i] for i in val_idx]
    X_test, y_test = [X[i] for i in test_idx], [y[i] for i in test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


print(split_data(X, y))
