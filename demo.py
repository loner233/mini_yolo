import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio

yolov0 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, (3,3), input_shape=(256, 256, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(12, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(24, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(36, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5)
])

optimizer = tf.optimizers.Adadelta(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
@tf.function
def train_step(x_batch, y_batch):
    print(x_batch)
    with tf.GradientTape() as tape:
        y_predict = yolov0.call(x_batch)
        loss = loss_fn(y_batch, y_predict)
    grads = tape.gradient(loss, yolov0.trainable_weights)
    optimizer.apply_gradients(zip(grads, yolov0.trainable_weights))
    return y_predict, loss

train_step(tf.random.normal(shape=(1, 256, 256, 3)), tf.Variable([[1, 2, 3, 4, 5]]))

def load_dataset():
    images, labels = [], []
    with open('./index.txt', 'r') as f:
        for line in f.readlines():
            splited_line = line.strip().split(",")
            uid = splited_line[0]
            x = float(splited_line[1]) / 256
            y = float(splited_line[2]) / 256
            w = float(splited_line[3]) / 256
            h = float(splited_line[4]) / 256
            c = int(splited_line[5])
            image = imageio.imread("pi/%s.png" % uid)
            images.append(image.tolist())
            labels.append([x, y, w, h, c])
    return images, labels

images, labels = load_dataset()
images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)

image_dataset = tf.data.Dataset.from_tensor_slices(images_tensor)
image_dataset = image_dataset.map(lambda x: x / 255.0)
y_dataset = tf.data.Dataset.from_tensor_slices(labels_tensor)
dataset = tf.data.Dataset.zip((image_dataset, y_dataset))

dataset = dataset.batch(64, drop_remainder=True)

tf.keras.losses.MeanSquaredError()(tf.convert_to_tensor([1]), tf.convert_to_tensor([2]))

for epoch in range(10):
    for x_batch, y_batch in dataset:
        y_predict, loss = train_step(x_batch, y_batch)
        print(loss)

