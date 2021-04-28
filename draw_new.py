def draw (epoch, hist, filepath = "./"):
    x_plot = []
    for i in range (1, epoch + 1):
        x_plot.append (i)

    plt.figure()
    plt.plot (x_plot, hist.history['loss'], label='train_loss')
    plt.scatter (x_plot, hist.history['loss'])
    plt.plot (x_plot, hist.history['val_loss'], label='val_loss')
    plt.scatter (x_plot, hist.history['val_loss'])
    plt.plot (x_plot, hist.history['accuracy'], label='train_acc')
    plt.scatter(x_plot, hist.history['accuracy'])
    plt.plot (x_plot, hist.history['val_accuracy'], label='val_acc')
    plt.scatter (x_plot, hist.history['val_accuracy'])
    plt.title ('Training Loss and Accuracy on Our_dataset')
    plt.xlabel ('Epoch #')
    plt.ylabel ('Loss/Accuracy')
    plt.legend ()
    plt.savefig (filepath + 'training.png')