from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.initializers import glorot_uniform


def create_model(grid):
    model = Sequential()
    model.add(LSTM(grid['LSTM_size'][1], input_shape=grid['input_shape'][1],
                   kernel_initializer=glorot_uniform(seed=grid['seed_model'][1]),
                   dropout=grid['dropout'][1]))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=grid['seed_model'][1])))
    return model
