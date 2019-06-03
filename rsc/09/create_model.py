from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.initializers import glorot_uniform
from keras import regularizers


def create_model(grid):
    model = Sequential()

    
    
    try:
        from keras.layers import CuDNNLSTM
        import google.colab
        from google.colab import drive
        model.add(CuDNNLSTM(grid['LSTM_size'][1], input_shape=grid['input_shape'][1],
                       kernel_initializer=glorot_uniform(seed=grid['seed_model'][1]),
                       dropout=grid['dropout'][1], kernel_regularizer=regularizers.l2(grid['kernel_regularizer'][1]),
                       recurrent_initializer='orthogonal'))
    except:
        model.add(LSTM(grid['LSTM_size'][1], input_shape=grid['input_shape'][1],
                       kernel_initializer=glorot_uniform(seed=grid['seed_model'][1]),
                       dropout=grid['dropout'][1], kernel_regularizer=regularizers.l2(grid['kernel_regularizer'][1]),
                       recurrent_initializer='orthogonal'))

    model.add(Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(seed=grid['seed_model'][1])))

    return model


