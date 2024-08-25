# Fashion MNIST Classification with Early Stopping

This script performs image classification on the Fashion MNIST dataset using TensorFlow and Keras, with an added feature to stop training early based on accuracy.

## Code Explanation

1. **Import Libraries**
    ```python
    import tensorflow as tf
    ```

2. **Define Custom Callback**
    ```python
    class callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > 0.95:
                print("\nEpoch %05d: stopping training due to high accuracy" % (epoch))
                self.model.stop_training = True
    ```
    - **Custom Callback**: Defines a callback that stops training when accuracy exceeds 95%.

3. **Instantiate Callback**
    ```python
    callbacks = callback()
    ```

4. **Load and Prepare Data**
    ```python
    data = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    ```
    - **Load Data**: Fetches the Fashion MNIST dataset, splitting it into training and testing sets.
    - **Normalize**: Scales pixel values to the range `[0, 1]` for better performance.

5. **Define the Model**
    ```python
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    ```
    - **Flatten**: Converts 2D image data into a 1D vector.
    - **Dense Layers**: Fully connected layers.
        - **ReLU Activation**: `tf.keras.layers.Dense(128, activation='relu')`
            - **ReLU (Rectified Linear Unit)**: A non-linear activation function that outputs the input directly if it is positive; otherwise, it outputs zero. This helps the model learn non-linear relationships.
        - **Softmax Activation**: `tf.keras.layers.Dense(10, activation='softmax')`
            - **Softmax**: Converts logits (raw prediction scores) into probabilities by normalizing the output across multiple classes, making it useful for multi-class classification problems.

6. **Compile the Model**
    ```python
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ```
    - **Optimizer**: `'adam'`
        - **Adam (Adaptive Moment Estimation)**: An adaptive learning rate optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent (SGD), namely Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
    - **Loss Function**: `'sparse_categorical_crossentropy'`
        - **Sparse Categorical Cross-Entropy**: Measures the difference between the true label and the predicted probability. It's used for multi-class classification where labels are integers.
    - **Metrics**: `['accuracy']`
        - **Accuracy**: The fraction of correctly classified samples. It's a common metric for classification tasks.

7. **Train the Model with Callback**
    ```python
    model.fit(train_images, train_labels, epochs=30, callbacks=[callbacks])
    ```
    - **Fit**: Trains the model for up to 30 epochs using the training data, with early stopping if accuracy exceeds 95%.

8. **Evaluate the Model**
    ```python
    model.evaluate(test_images, test_labels)
    ```
    - **Evaluate**: Assesses model performance on the test dataset.

## Summary

This script loads the Fashion MNIST dataset, preprocesses the images, defines a neural network model with an early stopping callback, compiles and trains the model, and finally evaluates its performance on the test data.
