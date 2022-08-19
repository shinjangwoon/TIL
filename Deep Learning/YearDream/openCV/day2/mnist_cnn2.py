import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values between 0~1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Configure a CNN model for MNIST
model = models.Sequential([
    layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(20, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(200, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training!
model.fit(train_images, train_labels, epochs=5)

# get model TF graph
tf_model_graph = tf.function(lambda x: model(x))

# get concrete function
tf_model_graph = tf_model_graph.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# obtain frozen concrete function
frozen_tf_func = convert_variables_to_constants_v2(tf_model_graph)

# get frozen graph
frozen_tf_func.graph.as_graph_def()

# save full tf model
tf.io.write_graph(graph_or_graph_def=frozen_tf_func.graph,
                  logdir="./frozen_models",
                  name="mnist_frozen_graph.pb",
                  as_text=False)
