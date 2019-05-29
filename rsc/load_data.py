from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_data(grid):
    right = pd.read_csv("../data/right.txt", header=None).values.tolist()
    left = pd.read_csv("../data/left.txt", header=None).values.tolist()
    
    BATCH_SIZE = grid['batch_size_data'][1]
    batch_right = [right[i:i + BATCH_SIZE] for i in range(0, len(right), BATCH_SIZE)]
    batch_left = [left[i:i + BATCH_SIZE] for i in range(0, len(left), BATCH_SIZE)]

    X_load = np.array(batch_right + batch_left)
    y_load = np.array([0]*len(batch_right) + [1]*len(batch_left))

    X_train, X_test, y_train, y_test = train_test_split(X_load, y_load, test_size=0.15, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test