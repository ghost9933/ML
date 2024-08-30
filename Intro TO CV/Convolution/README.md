## Convolutions

Convolutions are filters of weights that you use to multiply a pixel with its neighboring values. They are used to reduce the information in an image. For example, using `[[ -1, -1, -1 ], [ 0, 0, 0 ], [ 1, 1, 1 ]]` to extract horizontal lines from an image and `[[ -1, 0, 1 ], [ -1, 0, 1 ], [ -1, 0, 1 ]]` to extract vertical lines.

These filters help reduce images to a set of features. Neurons can then be used to identify the best features over time.

## Pooling

Pooling is a method to reduce the spatial dimensions of an image while retaining important information, often using max pooling.

By adding convolution layers with pooling layers to a neural network, we create a Convolutional Neural Network (CNN).



Here is a breakdown of the convolutional neural network (CNN) model:

```python
model = tf.keras.models.Sequential([
    # Convolutional layer with 64 filters of size 3x3, using ReLU activation function.
    # Input shape: 28x28 image with 1 channel (grayscale).
    # Output shape after convolution: 26x26x64 (64 feature maps).
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    
    # Max pooling layer with 2x2 pool size.
    # Output shape after pooling: 13x13x64.
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Another convolutional layer with 64 filters of size 3x3, using ReLU activation function.
    # Output shape after convolution: 11x11x64.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Max pooling layer with 2x2 pool size.
    # Output shape after pooling: 5x5x64.
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten layer to convert 3D matrix to 1D vector for dense layers.
    # Output shape after flattening: 1600 (5 * 5 * 64).
    tf.keras.layers.Flatten(),
    
    # Dense (fully connected) layer with 128 neurons and ReLU activation function.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Output layer with 10 neurons (corresponding to 10 classes in the dataset) and softmax activation function.
    tf.keras.layers.Dense(10, activation='softmax')
])
