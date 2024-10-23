import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical
from keras_tuner import BayesianOptimization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1_l2
from sklearn.utils import class_weight
from tensorflow.keras.applications import MobileNet

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Set the path to the directory containing the generated image data
# Use a relative path
path = os.path.join(os.path.dirname(__file__), 'generated_images')

# Set the list of classes
classes = ["Emily", "Michael", "Sophia", "Jacob"]
num_classes = len(classes)

# Load the image data
x_train = []
y_train = []
x_test = []
y_test = []
# Set the target size for resizing
target_size = (128, 128)

def pre_process():
    for i, c in enumerate(classes):
        class_path = os.path.join(path, c)
        images = [f for f in os.listdir(class_path) if os.path.isfile(
            os.path.join(class_path, f)) and f != '.DS_Store']
        num_train = int(len(images) * 0.7)
        num_test = len(images) - num_train
        for j, image in enumerate(images):
            img_path = os.path.join(class_path, image)
            img = Image.open(img_path)
            img = img.convert('RGB')  # Convert to RGB for MobileNet
            img = img.resize(target_size)
            img = np.array(img)
            img = img.astype('float32') / 255.0  # Normalize pixel values

            if j < num_train:
                x_train.append(img)
                y_train.append(i)
            else:
                x_test.append(img)
                y_test.append(i)

pre_process()

# Convert the data to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Compute class weights for imbalanced data using dictionary comprehension
class_weights = {i: weight for i, weight in enumerate(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train))}

# Preprocess the data
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest')

datagen.fit(x_train)

# Load the pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create the model
def build_model(hp):
    model = Sequential()

    # Add the pre-trained MobileNet model as a base
    model.add(base_model)

    model.add(Flatten())

    # Reshape the input to have three dimensions
    model.add(Reshape((1, -1)))

    # LSTM layers for character-level features
    model.add(Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=64, max_value=256, step=64),
                                 return_sequences=True)))
    model.add(Dropout(hp.Float('lstm_dropout', min_value=0.3, max_value=0.7, step=0.1)))

    # Dense layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'dense_units_{i}', min_value=64, max_value=256, step=64),
                        activation='relu',
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.3, max_value=0.7, step=0.1)))

    model.add(Flatten())  # Flatten the output before the final Dense layer

    model.add(Dense(num_classes, activation='softmax'))

    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, mode="min")

# Add callbacks to a list
callbacks_list = [reduce_lr, model_checkpoint, early_stopping]

# Initialize the tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='output2',
    project_name='Multi_Image_Processor_CNN')

# Search for the best model
tuner.search_space_summary()
tuner.search(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_test, y_test), class_weight=class_weights)

# Show the results
tuner.results_summary()

# Train the best model and plot the accuracy
def plotAndTrain():
    # Get the best models
    best_models = tuner.get_best_models(num_models=1)
    best_model = best_models[0]

    # Train the best model for 100 epochs
    history = best_model.fit(datagen.flow(x_train, y_train, batch_size=128),
                             validation_data=(x_test, y_test),
                             epochs=100,
                             callbacks=callbacks_list,
                             class_weight=class_weights)
    # Plot the accuracy history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

plotAndTrain()