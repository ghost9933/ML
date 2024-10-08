import tensorflow as tf

class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy')>0.95):
            print("\nEpoch %05d: stopping training due to high accuracy" % (epoch))
            self.model.stop_training = True

callbacks=callback()

data=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=data.load_data()
train_images=train_images/255.0
test_images=test_images/255.0
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=30,callbacks=[callbacks])

model.evaluate(test_images,test_labels)