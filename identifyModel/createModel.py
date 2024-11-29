from keras import Input, models, layers, optimizers


def createNewPredictionModel():
    inputs = Input((110, 110, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(152, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=x)

    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def createNewPredictionModel_v2():
    inputs = Input((110, 110, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(152, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=x)

    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def createNewPredictionModel_v3():
    inputs = Input((110, 110, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(152, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=x)

    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def createNewPredictionModel_vgg19():
    inputs = Input((110, 110, 1))

    # 1st Module
    conv1_1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1_2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv1_1)
    firstModule = layers.MaxPooling2D((2, 2), strides=2)(conv1_2)

    # 2nd Module
    conv2_1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(firstModule)
    conv2_2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2_1)
    secondModule = layers.MaxPooling2D((2, 2), strides=2)(conv2_2)

    # 3rd Module
    conv3_1 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        secondModule
    )
    conv3_2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3_1)
    conv3_3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3_2)
    thirdModule = layers.MaxPooling2D((2, 2), strides=2)(conv3_3)

    # 4th Module
    conv4_1 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(thirdModule)
    conv4_2 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4_1)
    conv4_3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4_2)
    fourthModule = layers.MaxPooling2D((2, 2), strides=2)(conv4_3)

    # 5th Module
    conv5_1 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(
        fourthModule
    )
    conv5_2 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv5_1)
    conv5_3 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv5_2)
    fifthModule = layers.MaxPooling2D((2, 2), strides=2)(conv5_3)

    # Fully Connected Layers
    flat = layers.Flatten()(fifthModule)
    fc1 = layers.Dense(1024, activation="relu")(flat)
    fc1 = layers.Dropout(0.5)(fc1)
    fc2 = layers.Dense(1024, activation="relu")(fc1)
    fc2 = layers.Dropout(0.5)(fc2)
    fc3 = layers.Dense(152, activation="softmax")(fc2)

    model = models.Model(inputs=inputs, outputs=fc3)

    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def createNewPredictionModel_vgg19():
    inputs = Input((110, 110, 1))

    # 1st Module
    conv1_1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1_2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(conv1_1)
    firstModule = layers.MaxPooling2D((2, 2), strides=2)(conv1_2)

    # 2nd Module
    conv2_1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(firstModule)
    conv2_2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(conv2_1)
    secondModule = layers.MaxPooling2D((2, 2), strides=2)(conv2_2)

    # 3rd Module
    conv3_1 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        secondModule
    )
    conv3_2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3_1)
    conv3_3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(conv3_2)
    thirdModule = layers.MaxPooling2D((2, 2), strides=2)(conv3_3)

    # 4th Module
    conv4_1 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(thirdModule)
    conv4_2 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4_1)
    conv4_3 = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(conv4_2)
    fourthModule = layers.MaxPooling2D((2, 2), strides=2)(conv4_3)

    # 5th Module
    conv5_1 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(
        fourthModule
    )
    conv5_2 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv5_1)
    conv5_3 = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(conv5_2)
    fifthModule = layers.MaxPooling2D((2, 2), strides=2)(conv5_3)

    # Fully Connected Layers
    flat = layers.Flatten()(fifthModule)
    fc1 = layers.Dense(1024, activation="relu")(flat)
    fc1 = layers.Dropout(0.5)(fc1)
    fc2 = layers.Dense(1024, activation="relu")(fc1)
    fc2 = layers.Dropout(0.5)(fc2)
    fc3 = layers.Dense(152, activation="softmax")(fc2)

    model = models.Model(inputs=inputs, outputs=fc3)

    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def createNewPredictionModel_vgg11():
    inputs = Input((110, 110, 1))

    # 1st Module
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    # 2nd Module
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    # 3rd Module
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    # 4th Module
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    # 5th Module
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), strides=2)(x)

    # Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(152, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=x)

    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def createNewPredictionModel_vggStyle():
    inputs = Input((110, 110, 1))

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dense(152, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=x)

    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
