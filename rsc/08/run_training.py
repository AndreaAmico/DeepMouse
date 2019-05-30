import time
import numpy as np
import tensorflow as tf
import random

from create_model import create_model
from helper import add_grid_and_save
from train_model import train_model

def run_training(X, y, grid, verbose=True):
    initial_time = time.time()
    np.random.seed(grid['seed_numpy'][1])
    tf.set_random_seed(grid['seed_tensorflow'][1])
    random.seed(grid['seed_random'][1])

    grid['input_shape'][1] = X.shape[1:]
    
    
    accuracies = []
    for index_split in range(grid['skf_n_splits'][1]): # looping on uncompleted CV trainings
        model = create_model(grid)
        print(f"\nCV validation {index_split+1} of {grid['skf_n_splits'][1]}")
        best_model_path = '{}/models/{}/best_model_{}_{}.pkl'.format(grid['root_path'][1], grid['version'][1], grid['test_index'][1], index_split)

        grid['best_model_paths'][1].append(best_model_path)
        grid = train_model(X, y, model, grid)
        grid['best_model_accuracies'][1].append(np.max(grid['fit_outs'][1][-1].history['val_acc']))
    
    add_grid_and_save(grid)

    total_time = time.strftime("%H hours, %M min, %S sec", time.gmtime((time.time() - initial_time)))
    print('  --  Model trained in {}'.format(total_time))
    return grid
