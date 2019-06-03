from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import LSTM
from keras.layers import GRU

from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras.initializers import glorot_uniform


def create_model(grid):
    model = Sequential()



    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))



    model.add(GRU(grid['GRU_size'][1],
                   kernel_initializer=glorot_uniform(seed=grid['seed_model'][1]),
                   dropout=grid['dropout'][1]))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=grid['seed_model'][1])))
    return model


# def create_model(grid):
#     model = Sequential()
    
#     model.add(GRU(grid['GRU_size'][1], input_shape=grid['input_shape'][1],
#                    kernel_initializer=glorot_uniform(seed=grid['seed_model'][1]),
#                    dropout=grid['dropout'][1]))
#     model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=grid['seed_model'][1])))
#     return model
