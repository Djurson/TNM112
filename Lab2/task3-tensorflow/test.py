from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model

import util
import data_generator

# --- CPU AND GPU LIMITING

tf.config.threading.set_inter_op_parallelism_threads(6) 
tf.config.threading.set_intra_op_parallelism_threads(6)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Använd Memory Growth (dynamisk allokering) ELLER minnesbegränsning, inte båda samtidigt.
        # Vi använder Memory Growth som ofta är enklast för att dela resurser:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth aktiverad.")
    except RuntimeError as e:
        print(e)

# ---

epochs = 40
batch_size = 128

# Stop if it isn't getting better after 6 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Load the PatchCamyleon dataset
# In this dataset, we don't have labels for the test set.
# Do your development by monitoring the validation performance,
# and when you are finished you will run predictions on the test
# set and produce a CSV file that you can upload to Kaggle.
data = data_generator.DataGenerator()
data.generate(dataset='patchcam')
data.plot()

keras.backend.clear_session()
# model = keras.Sequential()

data_augmentation = keras.Sequential([
    layers.RandomFlip(mode="horizontal_and_vertical"),
    layers.RandomRotation(factor=0.25),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomZoom(height_factor=(-0.15, 0.15)),
    layers.RandomShear(x_factor=0.05, y_factor=0.05),
    # layers.RandomColorJitter(brightness_factor=0.1, contrast_factor=0.2, saturation_factor=0.2, hue_factor=0.01),

    # layers.RandomFlip(mode="horizontal"),
    # layers.RandomFlip(mode="vertical"),

    # layers.RandomContrast(0.05),
    # layers.RandomBrightness(0.05),
])

inputs = keras.Input(shape=data.x_train.shape[1:])

x = data_augmentation(inputs)

residual_1 = x
residual_1 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='glorot_normal')(residual_1)

x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal')(x)
x = layers.BatchNormalization()(x)

x = layers.Add()([x, residual_1])
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

residual_2 = x
residual_2 = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='glorot_normal')(residual_2)

x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal')(x)
x = layers.BatchNormalization()(x)

x = layers.Add()([x, residual_2])
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

residual_3 = x

residual_3 = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer='glorot_normal')(residual_3)
x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal')(x)
x = layers.BatchNormalization()(x)

x = layers.Add()([x, residual_3])
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.3)(x)

x = layers.GlobalAveragePooling2D()(x)

# x = layers.Flatten()(x)
x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001), kernel_initializer='glorot_normal')(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.1)(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(data.K, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# model.add(layers.Input(shape=data.x_train.shape[1:]))

# model.add(data_augmentation)

# model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.4))

# model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.4))

# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005), kernel_initializer='glorot_normal'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.4))
# model.add(layers.Dense(data.K, activation='softmax'))

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.0015, weight_decay=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy','AUC'])
log = model.fit(data.x_train, data.y_train_oh, batch_size=batch_size, epochs=epochs, validation_data=(data.x_valid, data.y_valid_oh), validation_freq=1, verbose=True, callbacks=[early_stop, reduce_lr])

util.evaluate(model, data)
util.plot_training(log)

util.pred_test(model, data, name='submissions/submission_7.csv')