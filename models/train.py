import json
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.preprocessing import dataset_utils

# Set corresponding data directories
SIGNAL_DATA_DIR = '/path_to_data/data_woodpecker'
NOISE_DATA_DIR = '/path_to_data/data_noise'

MODELS_DIR = 'models/'
MODEL_TF = MODELS_DIR + 'model'
MODEL_NO_QUANTIZATION_TF_LITE = MODELS_DIR + 'model_no_quant.tflite'
MODEL_TF_LITE = MODELS_DIR + 'model.tflite'
MODEL_TF_LITE_MICRO = MODELS_DIR + 'model.cc'
MODEL_CHECKPOINT_FILE = 'best_model.h5'

logger = logging.getLogger(__file__)

"""
Data Sets
"""


def list_files(folder):
    if folder is not None:
        return os.listdir(folder)
    else:
        return list()


def read_json_file(selected_folder, selected_file):
    full_path = selected_folder + os.sep + selected_file
    try:
        with open(full_path, 'r') as reader:
            json_str = reader.read()
            json_obj = json.loads(json_str)
    except (IOError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(str(e))
        return None
    else:
        return json_obj


def label_data():
    signal_files = list_files(SIGNAL_DATA_DIR)
    noise_files = list_files(NOISE_DATA_DIR)
    all_files_index = list()
    for f in signal_files:
        all_files_index.append({'file': f, 'dir': SIGNAL_DATA_DIR, 'label': 'signal'})
    for f in noise_files:
        all_files_index.append({'file': f, 'dir': NOISE_DATA_DIR, 'label': 'noise'})

    random.shuffle(all_files_index)
    lbs = list()
    specs = list()
    global_max = -sys.maxsize
    global_min = sys.maxsize
    spec = None
    for f in all_files_index:
        jo = read_json_file(f['dir'], f['file'])
        if jo is None:
            continue

        spec = jo['spectrogram']
        specs.append(spec)
        if f['label'] == 'signal':
            lbs.append(1)
        elif f['label'] == 'noise':
            lbs.append(0)

        current_max = np.amax(spec)
        current_min = np.amin(spec)
        if current_max > global_max:
            global_max = current_max
        if current_min < global_min:
            global_min = current_min

    n_rows_time = len(spec)
    n_cols_freq = len(spec[0])
    n_classes = 2  # Two classes - signal of interest or everything else
    return specs, lbs, n_rows_time, n_cols_freq, n_classes, global_max, global_min


def build_training_datasets():
    seed = 123
    batch_size = 8

    specs, lbs, n_rows_time, n_cols_freq, n_classes, specs_max, specs_min = label_data()

    train_specs, train_labels = dataset_utils.get_training_or_validation_split(
        specs, lbs, 0.5, 'training')

    validation_specs, validation_labels = dataset_utils.get_training_or_validation_split(
        specs, lbs, 0.5, 'validation')

    train_specs_dataset = Dataset.from_tensor_slices(train_specs)
    train_labels_dataset = dataset_utils.labels_to_dataset(train_labels, 'binary', None)

    validation_specs_dataset = Dataset.from_tensor_slices(validation_specs)
    validation_labels_dataset = dataset_utils.labels_to_dataset(validation_labels, 'binary', None)

    train_zipped_dataset = dataset_ops.Dataset.zip((train_specs_dataset, train_labels_dataset))
    validation_zipped_dataset = dataset_ops.Dataset.zip((validation_specs_dataset, validation_labels_dataset))

    train_zipped_dataset = train_zipped_dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    train_zipped_dataset = train_zipped_dataset.batch(batch_size)

    validation_zipped_dataset = validation_zipped_dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    validation_zipped_dataset = validation_zipped_dataset.batch(batch_size)

    return train_zipped_dataset, validation_zipped_dataset, n_rows_time, n_cols_freq, n_classes, specs_max, train_specs


def build_testing_datasets():
    batch_size = 1
    specs, labels, n_rows_time, n_cols_freq, n_classes, specs_max, specs_min = label_data()

    test_specs_dataset = Dataset.from_tensor_slices(specs)
    test_labels_dataset = dataset_utils.labels_to_dataset(labels, 'binary', None)

    test_zipped_dataset = dataset_ops.Dataset.zip((test_specs_dataset, test_labels_dataset))
    test_zipped_dataset = test_zipped_dataset.batch(batch_size)

    return test_zipped_dataset, n_rows_time, n_cols_freq, n_classes, specs_max


"""
Model
"""


def define_model(height_n, width_n, n_classes, rescale_factor):
    image_channels = 1
    kernel_size = (3, 3)
    pool_size = (3, 3)

    mod = Sequential([
        layers.experimental.preprocessing.Rescaling(
            1. / rescale_factor, input_shape=(height_n, width_n, image_channels)
        ),
        layers.Conv2D(4, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size, padding='same'),
        layers.Conv2D(8, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size, padding='same'),
        layers.Conv2D(16, kernel_size, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size, padding='same'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

    mod.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    mod.summary()
    return mod


def create_tf_lite_model(rep_data):
    conv = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Using float, int8  quantization left for optimization step
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32

    def representative_dataset_gen():
        tmp = np.expand_dims(rep_data, axis=(-1))
        for data in tf.data.Dataset.from_tensor_slices(tmp).batch(1).take(50):
            rv = tf.dtypes.cast(data, tf.float32)
            yield [rv]

    conv.representative_dataset = representative_dataset_gen

    conv.experimental_new_converter = True
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    model_lite = conv.convert()
    open(MODEL_TF_LITE, "wb").write(model_lite)

    interp = tf.lite.Interpreter(model_content=model_lite)
    in_type = interp.get_input_details()[0]['dtype']
    print('input: ', in_type)
    out_type = interp.get_output_details()[0]['dtype']
    print('output: ', out_type)


"""
Standard model conversion to C for running on MCU
"""


def convert_to_c_srs():
    # Convert to a C source file, i.e, a TensorFlow Lite for Microcontrollers model
    cmd = 'xxd -i ' + MODEL_TF_LITE + ' > ' + MODEL_TF_LITE_MICRO
    os.system(cmd)
    # Update variable names
    replace_text = MODEL_TF_LITE.replace('/', '_').replace('.', '_')
    cmd = "sed -i 's/'" + replace_text + "'/g_model/g'" + MODEL_TF_LITE_MICRO
    os.system(cmd)


def plot_training_process(epochs_rng, train_acc, valid_acc, train_loss, valid_loss):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_rng, train_acc, label='Training Accuracy')
    plt.plot(epochs_rng, valid_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_rng, train_loss, label='Training Loss')
    plt.plot(epochs_rng, valid_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    train_ds, validation_ds, sp_height, sp_width, num_classes, specs_max_value, representative_specs_ds = \
        build_training_datasets()

    print("Dataset Parameters:")
    print("Max Amplitude: " + str(specs_max_value))
    element = train_ds.as_numpy_iterator().next()
    print("Shape: ", element[0].shape, element[1].shape)

    model = define_model(sp_height, sp_width, num_classes, specs_max_value)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs
    )

    model.save(MODEL_TF)

    acc = history.history['accuracy']
    validation_acc = history.history['val_accuracy']

    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plot_training_process(epochs_range, acc, validation_acc, loss, validation_loss)

    create_tf_lite_model(representative_specs_ds)
    convert_to_c_srs()
