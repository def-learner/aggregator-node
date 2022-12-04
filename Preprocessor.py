import tensorflow as tf
import numpy as np

test_data_dir = "InputData/"

IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 1
SEED = 1234567890 #SOMETHING RANDOM HERE
input_shape = (IMG_SIZE,IMG_SIZE, N_CHANNELS) # SHOULD BE IMG_HEIGHT and IMG_WIDTH but lets see

# Function to load image and return a dictionary
def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    print(image)

    # Three types of img paths: um, umm, uu
    # gt image paths: um_road, umm_road, uu_road
    mask_path = tf.strings.regex_replace(img_path, "image_2", "gt_image_2")
    mask_path = tf.strings.regex_replace(mask_path, "um_", "um_road_")
    mask_path = tf.strings.regex_replace(mask_path, "umm_", "umm_road_")
    mask_path = tf.strings.regex_replace(mask_path, "uu_", "uu_road_")
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    
    non_road_label = np.array([255, 0, 0])
    road_label = np.array([255, 0, 255])
    other_road_label = np.array([0, 0, 0])
    
    # Convert to mask to binary mask
    mask = tf.experimental.numpy.all(mask == road_label, axis = 2)
    mask = tf.cast(mask, tf.uint8)
    mask = tf.expand_dims(mask, axis=-1)

    return {'image': image, 'segmentation_mask': mask}

test_dataset = tf.data.Dataset.list_files(test_data_dir + "*.png", seed=SEED)
test_dataset = test_dataset.map(parse_image)

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

# Tensorflow function to preprocess validation images
@tf.function
def load_image_test(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

BATCH_SIZE = 32

#-- Testing Dataset --#
test_dataset['test'] = test_dataset['test'].map(load_image_test)
test_dataset['test'] = test_dataset['test'].batch(BATCH_SIZE)
test_dataset['test'] = test_dataset['test'].prefetch(buffer_size=tf.data.AUTOTUNE)