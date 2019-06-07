from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_data(grid):
    right_file_list = [f'{grid["root_path"][1]}data/r{index}.txt' for index in [1,2,3,5,6,7]]
    left_file_list = [f'{grid["root_path"][1]}data/l{index}.txt' for index in [1,2,3,5,6,7]]


    BATCH_SIZE = grid['batch_size_data'][1]

    right = []
    for right_file in right_file_list:
        right = right + pd.read_csv(right_file, header=None).values.tolist()

    left = []
    for left_file in left_file_list:
        left = left + pd.read_csv(left_file, header=None).values.tolist()


        
     
    # cut dataset to match an integer multiple of the batch size
    
    if len(right)%BATCH_SIZE != 0:
       right = right[:-(len(right)%BATCH_SIZE)]
    if len(left)%BATCH_SIZE != 0:
        left = left[:-(len(left)%BATCH_SIZE)]


    # split batches
    batch_right = [right[i:i + BATCH_SIZE] for i in range(0, len(right), BATCH_SIZE)]
    batch_left = [left[i:i + BATCH_SIZE] for i in range(0, len(left), BATCH_SIZE)]

    X_load = np.array(batch_right + batch_left)
    y_load = np.array([0]*len(batch_right) + [1]*len(batch_left))

    X_train, X_test, y_train, y_test = train_test_split(X_load, y_load, test_size=0.1, random_state=42, shuffle=True)

    if X_train.shape[0] > grid['training_size'][1]:
        X_train = X_train[:grid['training_size'][1], ...]
        y_train = y_train[:grid['training_size'][1]]
    return X_train, X_test, y_train, y_test
