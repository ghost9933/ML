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
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=30,callbacks=[callbacks])

model.evaluate(test_images,test_labels)

clasification=model.predict(test_images)
print(clasification[0])
print(test_labels[0])