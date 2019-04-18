# DEEP MOUSE

## Initial structure
- Problem definition
- Identification of required data
- Data pre-processing
- Definition of the training, dev and test set
- Algorithm selection
- Training
- Evaluation with test set


### Problem definition - 01
The goal of the algorithm is to identify if the laptop track-pad is used either with the right or the left hand.

### Identification of required data - 01
Mouse position in time. List of x-y coordinates giving the absolute position of the mouse every \~10ms. The data are acquired for about 10 minutes by using the track-pad with the right hand while reading a technical blog post. The same is repeated by using the left hand. The total amount of samples is 120k (60k right, 60k left).
The data are acquired using the win32gui library and stored in a *.txt* file via a python script:
```python
import win32gui, time
with open('./data/my_data.txt', 'a') as f:
    for _ in range(number_of_samples):
        x, y = win32gui.GetCursorPos()
        f.write('{},{}\n'.format(x, y))
        time.sleep(0.01)
```
Right hand data are saved on *./data/right.txt* file, left hand data on *./data/left.txt* file.
As shown in the plot below the real time delay between subsequent acquisitions is not completely constant, moreover spikes are present.

![Data acquisition stability](./plots/data_acquisition_stability.png)

A possible improvement might be achieved by a pure c acquisition program, which includes a time delay check every loop.

### Data pre-processing - 01
The full 20 minutes dataset is split in batches of 200 points each, corresponding to 2 seconds of mouse position acquisitions. Right and left hand data are merged together in a single dataset. For the moment we use raw data from the input device, being the absolute coordinate along the horizontal and vertical direction of the screen.

---------------
Load of the mouse data from the *.txt* file using **pandas**:
```python
import pandas as pandas
right = pd.read_csv("../data/right.txt", header=None).values.tolist()
left = pd.read_csv("../data/left.txt", header=None).values.tolist()
```
Splitting the data in 600 batches containing 200 data-points each:
```python
batch_size = 200
batch_right = [right[i:i + batch_size] for i in range(0, len(right), batch_size)]
batch_left = [left[i:i + batch_size] for i in range(0, len(left), batch_size)]
```
Merging left and right datasets and convert them into **numpy** arrays. Create the target array `y` using as convention *0* for batches corresponding tho right hand and *1* for left hand. The axis of the `X` array correspond to: (batch index, mouse position in time, mouse coordinate index).

```python
import numpy as np
X = np.array(batch_right + batch_left)
y = np.array([0]*len(batch_right) + [1]*len(batch_left))

print(f'X shape: {X.shape}\ny shape: {y.shape}')
```
```text
X shape: (600, 200, 2)
y shape: (600,)
```

In the plot below we show the data contained in a single data batch.
![Batch data example](./plots/batch_example.png)


### Definition of the training set - 01
**Training**, **dev** and **test** sets are split in a 70%-15%-15% proportion. The training set is used for training the network, the dev set as a benchmark to optimize the ML algorithm and finally the test set to measure the accuracy of the model. It is important to keep dev and test set separated to avoid the over-fitting of the hyper-parameters of the model on the test set. 

------------
The splitting between train/dev/test is achieved using the **sklearn** library.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
X_dev, X_test, y_dev, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=1)
```

### Algorithm selection
As a starting point algorithm we opted for a **RNN** (recurrent neural network). In particular, inspired from this [blog post](https://www.analyticsvidhya.com/blog/2019/01/introduction-time-series-classification/#), we used a **LSTM** (Long short-term memory) architecture.

-------------------
We implemented the neural network in **keras** using **TensorFlow backend**:
```python
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(256, input_shape=(batch_size, 2)))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```
```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 256)               265216    
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 257       
=================================================================
Total params: 265,473
Trainable params: 265,473
Non-trainable params: 0
_________________________________________________________________
```

The training is performed using **stochastic gradient descent**, in particular using the **Adam** algorithm (short for Adaptive Moment Estimation). We used the *accuracy* metric and we trained the data for 200 epochs. We save the best model as *best_model.pkl*. It takes about 8 minutes to train the model using a regular laptop.
```python
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

adam = Adam(lr=0.001)
chk = ModelCheckpoint('best_model.pkl', monitor='acc', save_best_only=True, mode='max', verbose=0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=64, callbacks=[chk], validation_data=(X_dev, y_dev))
```
The learning process during the gradient descent can be visualized by monitoring the accuracy of the model and the loss function, computed both on the training set and on the dev set.

![Model 01](./plots/model_01.png)

The accuracy describe the ratio between successful and total guesses. The loss-function correspond to the **binary cross-entropy**, which is given by:
```text
-(y log(p) + (1-y) log(1-p))
```
where `y` is the target correct binary label (0 for right hand, 1 for left hand) and `p` is the predicted probability for a given data batch to be a left hand batch. When the cross-entropy is *1* the model is useless and it is equivalent to a random guess. When it is *0* the model perfectly predict the target given a single data batch.

From the plot we can observe an accuracy increase both for the training set and the dev set, but the slope is very slow. The loss function decreases as well and it seems not to be saturated after 200 epochs of training. Maybe more training time might be beneficial. The best accuracy achieved on the dev set is of about *0.78* which is a promising starting point.

### Evaluation of the model
<!-- We 
from keras.models import load_model -->