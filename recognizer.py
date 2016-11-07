# pylint: disable=I0011, line-too-long, C0103, C0111
import matplotlib.pyplot as plt
import skimage as skimage
import numpy as np
from numpy.random import RandomState
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
        random_state = RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def predict_ensemble(ensemble_models, input_data, batch_size):
    models_num = len(ensemble_models)
    ensemble_predictions = np.empty((models_num, input_data.shape[0], 10))
    for model_idx in range(0, models_num):
        ensemble_model = ensemble_models[model_idx][0]
        ensemble_predictions[model_idx] = ensemble_model.predict(input_data, batch_size=batch_size, verbose=1)
    return ensemble_predictions.mean(axis=0)

def create_model(print_summary=True):
    model = Sequential()

    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))

    if print_summary:
        model.summary()
    return model

# want to see layers
create_model(True)

print('Loading training data...')

# train_data_set = np.loadtxt('train_1000.csv', delimiter=',', skiprows=1, dtype=int)
train_data_set = np.loadtxt('train.csv', delimiter=',', skiprows=1, dtype=int)

print('Loading test data...')

# predict_data_set = np.loadtxt('test_1000.csv', delimiter=',', skiprows=1, dtype=int)
predict_data_set = np.loadtxt('test.csv', delimiter=',', skiprows=1, dtype=int)
X_predict = predict_data_set.reshape((predict_data_set.shape[0], 1, 28, 28)).astype('float') / 255

print('All data loaded!')

ensemble_models_num = 5
batch_size = 64
models = []

# build an ensemble of models
for model_idx in range(0, ensemble_models_num):

    np.random.shuffle(train_data_set)

    model = create_model(False)

    Y = to_categorical(train_data_set[:, 0], 10)
    X = train_data_set[:, 1:].reshape((train_data_set.shape[0], 1, 28, 28)).astype('float') / 255

    print('Generating additional training data for model ', model_idx + 1)

    total_items_num = X.shape[0]
    train_items_num = int(0.8 * total_items_num)
    training_set_multiplier = 10
    Y_train = Y[0:train_items_num].repeat(training_set_multiplier, axis=0)
    X_train = X[0:train_items_num].repeat(training_set_multiplier, axis=0)
    for i in range(train_items_num):
        for j in range(1, training_set_multiplier):
            x = X_train[i * training_set_multiplier]
            # TODO Get rid of double re-shape
            X_train[i * training_set_multiplier + j] = skimage.util.random_noise(elastic_transform(x.reshape((28, 28)), 28, np.random.uniform(28 * 0.15, 28 * 0.15)), 's&p').reshape((1, 28, 28))
    Y_val = Y[train_items_num:]
    X_val = X[train_items_num:]

    # plt.figure(figsize=(28, 28))
    # for i in range(100):
    #     plt.subplot(10, 10, i + 1)
    #     plt.imshow(X_train[i].reshape((28, 28)), cmap='gray')
    #     plt.text(0.5, 0.5, str(i), color='white')
    #     plt.axis('off')
    # plt.show()

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    print('Training model...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=50, verbose=1, validation_data=(X_val, Y_val), callbacks=[early_stopping])

    score = model.evaluate(X_val, Y_val, verbose=1, batch_size=batch_size)
    print('Test score:', score)

    with open('model-' + str(1 - score) + '.json', 'w') as model_file:
        model_file.write(model.to_json())

    models.append([model, score])

# predict test data 
predictions = predict_ensemble(models, X_predict, batch_size)
class_predictions = predictions.argmax(axis=1)

np.savetxt('submission.csv', np.stack((np.arange(class_predictions.shape[0]) + 1, class_predictions), axis = -1), header='ImageId,Label', delimiter=',', fmt='%1i', comments='')

# review misidentified numbers from validation set
Y = to_categorical(train_data_set[:, 0], 10)
X = train_data_set[:, 1:].reshape((train_data_set.shape[0], 1, 28, 28)).astype('float') / 255

predictions = predict_ensemble(models, X, batch_size)
class_predictions = predictions.argmax(axis=1)
Y_class = Y.argmax(axis=1)

plt.figure(figsize=(28, 28))
img_idx = 1
for i in range(X.shape[0]):
    y_pre = class_predictions[i]
    if Y_class[i] != y_pre and img_idx <= 100:
        plt.subplot(10, 10, img_idx)
        plt.imshow(X[i].reshape((28, 28)), cmap='gray')
        plt.text(0.5, 0.5, str(y_pre), color='white')
        plt.axis('off')
        img_idx += 1
plt.show()
        