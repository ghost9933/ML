# Fashion MNIST Classification with TensorFlow

This script performs image classification on the Fashion MNIST dataset using TensorFlow and Keras. Below is a step-by-step explanation:

## Code Explanation

1. **Import Libraries**
    ```python
    import tensorflow as tf
    ```

2. **Load and Prepare Data**
    ```python
    data = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    ```
    - **Load Data**: Fetches the Fashion MNIST dataset, splitting it into training and testing sets.
    - **Normalize**: Scales pixel values to the range `[0, 1]` for better performance.

3. **Define the Model**
    ```python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    ```
    - **Flatten**: Converts 2D image data into a 1D vector.
    - **Dense Layers**: Fully connected layers; the first with ReLU activation and the second with softmax for classification.

4. **Compile the Model**
    ```python
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ```
    - **Optimizer**: 'adam' for adaptive gradient descent.
    - **Loss Function**: 'sparse_categorical_crossentropy' for multi-class classification.
    - **Metrics**: 'accuracy' to evaluate model performance.

5. **Train the Model**
    ```python
    model.fit(train_images, train_labels, epochs=5)
    ```
    - **Fit**: Trains the model for 5 epochs using the training data.

## Summary

This script loads the Fashion MNIST dataset, preprocesses the images, defines a simple neural network model, compiles it with appropriate settings, and trains it on the training data.
