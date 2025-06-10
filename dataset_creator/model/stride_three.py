from tensorflow.keras import layers, models, Input, Model


def get_model(num_input_samples, num_classes):
    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=96,
            kernel_size=15,
            strides=3,
            activation="elu",
            input_shape=(num_input_samples, 1),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=96, kernel_size=13, strides=3, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=128, kernel_size=13, strides=3, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=128, kernel_size=11, strides=3, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=192, kernel_size=10, strides=3, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=192, kernel_size=8, strides=3, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=256, kernel_size=9, strides=3, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=256, kernel_size=8, strides=3, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=256, kernel_size=1, strides=1, activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(filters=128, kernel_size=1, strides=1, activation="linear"))
    #    model.add(layers.BatchNormalization())

    #    model.add(layers.Flatten())
    #    model.add(layers.Dense(units=num_classes, activation="sigmoid"))

    return model
