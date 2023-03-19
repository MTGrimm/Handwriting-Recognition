import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# getting dataset
mnist = tf.keras.datasets.mnist

# training data
# x_train -> image data
# y_train -> label 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # normalize
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
task = 0
while task != 3:
    task = int(input("1: Train model\n2: Test model\n3: End session\n= "))
    if task == 1:
        ep = int(input("Enter number of epochs used to train: "))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # flattens the input from 28x28 to a single 784 line
        model.add(tf.keras.layers.Dense(128, activation='exponential'))
        model.add(tf.keras.layers.Dense(128, activation='exponential'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))  # ouput layer, softmax -> makes all the 10 neurons add upto 1

        # compiling model
        model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # training model
        model.fit(x_train, y_train, epochs=ep) 

        # saving model
        model.save('handwritten.model')

    elif task == 2:
        model = tf.keras.models.load_model('Python\Handwritting Recognition\handwritten.model')

        image_number = 0
        while os.path.isfile(f"Python/Handwritting Recognition/digits/digit{image_number}.png"):
            try:
                img = cv2.imread(f"Python/Handwritting Recognition/digits/digit{image_number}.png")[:,:,0]
                img = np.invert(np.array([img]))
                prediction = model.predict(img)
                print(f"This digit is probably a {np.argmax(prediction)}")
                plt.imshow(img[0], cmap=plt.cm.binary)
                plt.show()
            except:
                print("Error!")
            finally:
                image_number += 1

        loss, accuracy = model.evaluate(x_test, y_test)

        print(loss)
        print(accuracy)