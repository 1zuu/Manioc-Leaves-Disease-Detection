import pathlib
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt

from util import *
from variables import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class ManiocDiseaseDetection(object):
    def __init__(self):
        train_generator, validation_generator = image_data_generator()
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.train_step = self.train_generator.samples // batch_size
        self.validation_step = self.validation_generator.samples // valid_size

        self.accuracy = tf.keras.metrics.BinaryAccuracy(threshold=thresholds)
        self.recall = tf.keras.metrics.Recall(thresholds=thresholds)
        self.precision = tf.keras.metrics.Precision(thresholds=thresholds)

    def classifier(self, x):
        if not self.trainable:
            x = Dense(dense_1, activation='relu')(x)
            x = Dense(dense_1)(x)
            x = BatchNormalization()(x)
            x = relu(x)
            x = Dropout(rate)(x)

            x = Dense(dense_2, activation='relu')(x)
            x = Dense(dense_2)(x)
            x = BatchNormalization()(x)
            x = relu(x)
            x = Dropout(rate)(x)

        x = Dense(dense_3, activation='relu')(x)
        x = Dense(dense_3)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(rate)(x)
        return x

    def model_conversion(self, trainable):
        functional_model = tf.keras.applications.MobileNetV2(weights="imagenet")
        functional_model.trainable = trainable

        self.trainable = trainable

        inputs = functional_model.input

        x = functional_model.layers[-2].output
        x = self.classifier(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(
                inputs =inputs,
                outputs=outputs
                    )
        self.model = model
        self.model.summary()

    def train(self):
        callback = tf.keras.callbacks.EarlyStopping(
                                                monitor='loss', 
                                                patience=3
                                                    )

        self.model.compile(
                          optimizer='Adam',
                          loss='binary_crossentropy',
                          metrics=[
                                self.accuracy,
                                self.recall,
                                self.precision
                                  ]
                          )
        self.model.fit(
                    self.train_generator,
                    steps_per_epoch= self.train_step,
                    validation_data= self.validation_generator,
                    validation_steps = self.validation_step,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=[callback]
                        )

    def save_model(self):
        self.model.save(model_weights)
        print("MobileNet Saved")

    def loading_model(self):
        K.clear_session() #clearing the keras session before load model
        self.model = load_model(model_weights)
        self.model.compile(
                          optimizer='Adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy']
                          )
        print("MobileNet Loaded")

    def TFconverter(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(model_converter)
        model_converter_file.write_bytes(tflite_model)

    def TFinterpreter(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_converter)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def Inference(self, img):
        input_shape = self.input_details[0]['shape']
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        assert np.array_equal(input_shape, input_data.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

    def run(self):
        if not os.path.exists(model_converter):
            if not os.path.exists(model_weights):
                self.model_conversion(False)
                self.train()
                self.save_model()
            else:
                self.loading_model()
            self.TFconverter()
        self.TFinterpreter()    

model = ManiocDiseaseDetection()
model.run()