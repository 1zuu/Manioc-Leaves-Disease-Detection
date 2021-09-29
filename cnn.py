import pathlib
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

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
        train_generator, validation_generator, test_generator = image_data_generator()
        self.test_generator = test_generator
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_step = self.test_generator.samples // batch_size
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

    def plot_confusion_matrix(self, data_generator, split, cmap=None, normalize=False):
        
        Y = []
        P = []
        for x, y in data_generator:
            pred = self.model.predict(x)
            p = pred > thresholds
            Y.extend(y)
            P.extend(p)
    
        Y = np.array([class_dict[y] for y in Y])
        P = np.array([class_dict[p] for p in P])
        cm = confusion_matrix(Y, P)

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(30, 30))
        plt.rcParams.update({'font.size': 22})
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix for Website Policy Classification')
        plt.colorbar()

        class_names = list(set(self.encoder.inverse_transform(Y)))

        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=0)
            plt.yticks(tick_marks, class_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),

                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix for {} Set'.format(split))
        plt.savefig(cm_path.format(split))

    def evaluations(self):
        self.model.evaluate(self.train_generator, steps=self.train_step)
        self.model.evaluate(self.validation_generator, steps=self.validation_step)
        self.model.evaluate(self.test_generator, steps=self.test_step)

    def visualization(self):
        self.plot_confusion_matrix(self.train_generator, 'train')
        self.plot_confusion_matrix(self.validation_generator, 'validation')
        self.plot_confusion_matrix(self.test_generator, 'test')

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
            self.evaluations()
            self.visualization()
            self.TFconverter()
        self.TFinterpreter()    

if __name__ == "__main__":
    model = ManiocDiseaseDetection()
    model.run()