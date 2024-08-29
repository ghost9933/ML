## Convolutions

Convolutions are filters of weights that you use to multiply a pixel with its neighboring values. They are used to reduce the information in an image. For example, using `[[ -1, -1, -1 ], [ 0, 0, 0 ], [ 1, 1, 1 ]]` to extract horizontal lines from an image and `[[ -1, 0, 1 ], [ -1, 0, 1 ], [ -1, 0, 1 ]]` to extract vertical lines.

These filters help reduce images to a set of features. Neurons can then be used to identify the best features over time.

## Pooling

Pooling is a method to reduce the spatial dimensions of an image while retaining important information, often using max pooling.

By adding convolution layers with pooling layers to a neural network, we create a Convolutional Neural Network (CNN).
