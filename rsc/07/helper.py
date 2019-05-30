import time
from keras.callbacks import Callback
import numpy as np


def progress_bar(current_value, max_value):
    progress = ((current_value+1)/max_value)*100
    if progress>98: progress=100
    print('\r[{0}{1}] {2:.1f}%'.format('#'*int(progress/2), ' '*(50-int(progress/2)), progress), end='')


def play_bell():
    import winsound
    duration = 200  # millisecond
    freq = 440  # Hz
    for i in range(5):
        winsound.Beep(int(freq*(i/2+1)), duration)
    

class LossHistory(Callback):
    def __init__(self, number_of_epochs):
        self.number_of_epochs = number_of_epochs
        self.current_epoch = 0
    def on_train_begin(self, logs={}):
        self.initial_time = time.time()
    def on_batch_end(self, batch, logs={}):
        if logs['batch']==0:
            NUMBER_OF_DIESIS = 20
            self.current_epoch += 1
            progress = self.current_epoch/self.number_of_epochs
            diesis = np.round(progress*NUMBER_OF_DIESIS).astype('int')
            eta = (time.time()-self.initial_time) * (self.number_of_epochs/self.current_epoch -1)
            remaining_time = time.strftime("%H hours, %M min, %S sec", time.gmtime(eta))
            print('\r[{}{}] acc: {:.3f} eta: {}'.format('#'*diesis, '-'*(NUMBER_OF_DIESIS-diesis), logs['acc'], remaining_time), end='')

def add_grid_and_save(grid):
    grid_file_path = '{}/model_data/grid_{}.pkl'.format(grid['root_path'][1], grid['version'][1])
    
    if os.path.isfile(grid_file_path):
        grid_df = pd.read_pickle(grid_file_path)
    else:
        grid_df = pd.DataFrame()

    grid_df = grid_df.append({key:grid[key][1] for key in grid.keys()}, ignore_index=True)        
    grid_df.to_pickle(grid_file_path)
