import matplotlib.pyplot as plt
import numpy as np


def plot_image(prediction_array, true_label, image, class_names):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='binary')

    predicted_label = np.argmax(prediction_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(prediction_array),
                                         class_names[true_label],
                                         ),
               color=color)


def plot_value_array(prediction_array, true_label):
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_loss_graph(h):
    # Plot training & validation loss values. Takes keras history object
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
