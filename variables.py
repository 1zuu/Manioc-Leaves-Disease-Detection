batch_size = 32
valid_size = 16
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.3
zoom_range = 0.3
rotation_range = 30
shift_range = 0.3
mean = 0
std = 255.0
dense_1 = 512
dense_2 = 256
dense_3 = 64
num_classes = 5
epochs = 10
rate = 0.2

verbose = 1
val_split = 0.2
seed = 1234

# data directories and model paths
train_dir = 'data/train_images/'
train_labels = 'data/train.csv'
model_weights = "data/weights/model_weights.h5"
model_converter = "data/weights/model.tflite"