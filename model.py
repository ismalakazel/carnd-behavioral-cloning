from datagen import generator, get_data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Model params
BATCH_SIZE = 32
OPTMIZER = Adam(0.0001)
LOSS = 'mse'
INPUT_SHAPE = (66, 200, 3)
NB_EPOCHS = 100


# NVIDIA model
# source: https://arxiv.org/pdf/1604.07316.pdf
model = Sequential()
model.add(Lambda(lambda x: x/127.5 -1, input_shape=INPUT_SHAPE))
model.add(Conv2D(24,5,5,border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Conv2D(36,5,5,border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Conv2D(48,5,5,border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Conv2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1)))
model.add(Conv2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.compile(loss=LOSS, optimizer=OPTMIZER)
model.summary()


# Read cvs from data/ folder
images, labels = get_data('./data/driving_log.csv')


# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.2)


# Define generators
train_generator = generator(X_train, y_train, batch_size=BATCH_SIZE, training=True)
validation_generator = generator(X_valid, y_valid, batch_size=BATCH_SIZE)


# Define early stopping and checkpoint callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    mode='auto')

checkpoint = ModelCheckpoint(
    './weights/weights-{epoch:02d}-{val_loss:.4f}.hdf5',
    monitor='val_loss',
    save_best_only=True,
    mode='min')


# Train and save model
model.fit_generator(
    train_generator,
    samples_per_epoch=len(X_train) // 12,
    validation_data=validation_generator,
    nb_val_samples=len(X_valid),
    nb_epoch=NB_EPOCHS,
    callbacks=[early_stopping, checkpoint])
model.save('model.h5')
