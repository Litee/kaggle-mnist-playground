import matplotlib.pyplot as plt
import numpy as np
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
   
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

print 'Loading training data...'

# train_data_set = np.loadtxt('debug.csv', delimiter=',', skiprows=1, dtype=int)
train_data_set = np.loadtxt('train.csv', delimiter=',', skiprows=1, dtype=int)
Y = to_categorical(train_data_set[:, 0], 10)
X = train_data_set[:,1:].reshape((train_data_set.shape[0], 1, 28, 28)).astype('float') / 255

print 'Generating additional training data...'

items_num = X.shape[0]
train_items_num = int(0.95 * items_num) 
training_set_multiplier = 10
Y_train = Y[0:train_items_num].repeat(training_set_multiplier, axis=0)
X_train = X[0:train_items_num].repeat(training_set_multiplier, axis=0)
for i in range(train_items_num):
    for j in range(1, training_set_multiplier):
        x = X_train[i * training_set_multiplier]
        # TODO Get rid of double re-shape
        X_train[i * training_set_multiplier + j] = elastic_transform(x.reshape((28, 28)), 28, np.random.uniform(28 * 0.15, 28 * 0.15)).reshape((1, 28, 28))
Y_val = Y[train_items_num:]
X_val = X[train_items_num:]

print 'Loading test data...'

# predict_data_set = np.loadtxt('debug_test.csv', delimiter=',', skiprows=1, dtype=int)
predict_data_set = np.loadtxt('test.csv', delimiter=',', skiprows=1, dtype=int)
X_predict = predict_data_set.reshape((predict_data_set.shape[0], 1, 28, 28)).astype('float') / 255

print('All data loaded!')

# plt.figure(figsize=(28, 28))
# plt.imshow(X_train[3], cmap='gray')
# plt.show()

# print Y_train[0:25]
# plt.figure(figsize=(28, 28))
# for i in range(100):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(X_train[i].reshape((28, 28)), cmap='gray')
#     plt.text(0.5, 0.5, str(i), color='white')
#     plt.axis('off')
# plt.show()

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 28, 28), activation='relu')) 
model.add(Dropout(0.3))

model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2))) 
model.add(Dropout(0.3))

model.add(Convolution2D(128, 3, 3, activation='relu', subsample=(2, 2))) 
model.add(Dropout(0.3))

model.add(Convolution2D(256, 3, 3, activation='relu')) 
model.add(Dropout(0.3))

model.add(Convolution2D(512, 3, 3, activation='relu')) 
model.add(Dropout(0.3))

model.add(Convolution2D(128, 1, 1, activation='relu')) 
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

batch_size = 32
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=50, verbose=1, validation_data=(X_val, Y_val), callbacks=[early_stopping])

score = model.evaluate(X_val, Y_val, verbose=1, batch_size=batch_size)
print('Test score:', score)

predictions = model.predict(X_predict, batch_size=batch_size, verbose=1)
class_predictions = predictions.argmax(axis=1)

np.savetxt('submission.csv', np.stack((np.arange(class_predictions.shape[0]) + 1, class_predictions), axis = -1), header='ImageId,Label', delimiter=',', fmt='%1i', comments='')

predictions = model.predict(X_val, batch_size=batch_size, verbose=1)
class_predictions = predictions.argmax(axis=1)
Y_val_class = Y_val.argmax(axis=1)

plt.figure(figsize=(28, 28))
img_idx = 1
for i in range(X_val.shape[0]):
    y_pre = class_predictions[i]
    if Y_val_class[i] != y_pre and img_idx <= 100:
        plt.subplot(10, 10, img_idx)
        plt.imshow(X_val[i].reshape((28, 28)), cmap='gray')
        plt.text(0.5, 0.5, str(y_pre), color='white')
        plt.axis('off')
        img_idx += 1
plt.show()
        