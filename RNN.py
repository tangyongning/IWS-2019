from pandas import read_csv
from pandas import datetime
from pandas import concat
import numpy as np
from keras.layers import SimpleRNN, Dense, LSTM, Bidirectional, GRU
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" 


def parser(x):
    if len(x) == 8:
        x = '0' + x
    return datetime.strptime(x, '%d-%B-%y')


def table2lags(table, max_lag, min_lag=0, separator='_'):
    values = []
    for i in range(min_lag, max_lag + 1):
        values.append(table.shift(i).copy())
        values[-1].columns = [c + separator + str(i) for c in table.columns]
    return concat(values, axis=1)


series = read_csv('Kirie_Edited_Data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
lag = 6
num_features = 6

series_normalized = (series-np.mean(series))/np.std(series)
#series_normalized = (series - np.min(series))/(np.max(series)-np.min(series))

X = table2lags(series_normalized, lag-1)

name = 'NH3_0'
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
test_target = test[name]
train_target = train[name][lag:]
train = train[lag - 1:]

X_train = train[:-1].values.reshape(-1,lag,num_features).astype('float32')
y_train = train_target.values.astype('float32')

X_test = test.values.reshape(-1,lag,num_features).astype('float32')
y_test = test_target.values.astype('float32')

hidden = 64
batch_size = 20
epochs = 15

model = Sequential()
model.add(LSTM(hidden, input_shape=(lag, num_features)))
#model.add(SimpleRNN(hidden))
model.add(Dense(1))


#Following four lines are for model saving

'''
model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test,y_test),
		    callbacks=callbacks_list, verbose=False)
'''


model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test,y_test))

y_predict = model.predict(X_test)

#model.save_weights(name+'_'+str(epochs)+'.h5')



plt.plot(y_test)
plt.plot(y_predict, color='red')
plt.show()
