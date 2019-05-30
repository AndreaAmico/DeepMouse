import time

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from helper import play_bell, LossHistory



def train_model(X, y, model, grid):
    skf = StratifiedKFold(n_splits=grid['skf_n_splits'][1], shuffle=True, random_state=grid['seed_skf'][1])
    train_index, dev_index = list(skf.split(X, y))[len(grid['fit_outs'][1])]
    X_train, X_dev = X[train_index], X[dev_index]
    y_train, y_dev = y[train_index], y[dev_index]

    adam = Adam(lr=grid['learning_rate'][1])
    progress_bar = LossHistory(grid['epochs'][1])
    
    chk = ModelCheckpoint(grid['best_model_paths'][1][-1], monitor=grid['best_model_metric'][1],
                          save_best_only=True, mode='max', verbose=0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[grid['training_metric'][1]])
    fit_out = model.fit(X_train, y_train, epochs=grid['epochs'][1], batch_size=grid['batch_size'][1],
                        callbacks=[chk, progress_bar], validation_data=(X_dev, y_dev), verbose=0)

    fit_out.model = None
    grid['fit_outs'][1].append(fit_out)
    return grid