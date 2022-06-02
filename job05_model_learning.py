import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

x_train,x_test,y_train,y_test=np.load(
    './crawling_data/news_data_max_17_wordsize_12426.npy',
        allow_pickle=True)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

model=Sequential() #단어 개수의 차원
model.add(Embedding(12426,300,input_length=17)) #12426차원을 300차원으로 축소
model.add(Conv1D(32,kernel_size=5, padding='same',activation='relu'))
model.add(MaxPool1D(pool_size=1)) #pooling 안함
model.add(LSTM(128,activation='tanh',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64,activation='tanh',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64,activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
fit_hist=model.fit(x_train,y_train,batch_size=128,
                   epochs=10,validation_data=(x_test,y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'],label='val_accuracy')
plt.plot(fit_hist.history['accuracy'],label='accuracy')
plt.legend()
plt.show()