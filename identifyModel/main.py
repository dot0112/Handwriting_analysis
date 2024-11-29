import sys
import os
import datetime

argv = sys.argv
scriptFileName = os.path.basename(argv[0])
currDir = os.path.dirname(os.path.realpath(__file__))

print()
print("Usage:")
print("     python %s train [<epochs>]" % scriptFileName)
print("     python %s test" % scriptFileName)
print()

trainEpochs = 10000
if argv[1] == "train":
    isTrain = True
    if len(argv) >= 3:
        trainEpochs = int(argv[2])

elif argv[1] == "test":
    isTrain = False

else:
    raise ValueError("Invalid mode %s" % argv[1])


import keras
import createModel
import createDataset
import getTrainData
import getValData

modelPath = os.path.join(currDir, "identifyModel.keras")

if isTrain:
    model = None
    if os.path.isfile(modelPath):
        print("Loading existing model")
        loadedModel = keras.models.load_model(modelPath)
        model = createModel.createNewPredictionModel()
        model.set_weights(loadedModel.get_weights())
    else:
        print("Creating new model")
        model = createModel.createNewPredictionModel()

    print("****************************************")
    print("Start training for %g epochs" % trainEpochs)

    def getCallbacks():
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=modelPath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        )

        class DisplayTimeCallback(keras.callbacks.Callback):
            def __init__(
                self,
            ):
                super(DisplayTimeCallback, self).__init__()

            def on_epoch_begin(self, epoch, logs=None):
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        display_time = DisplayTimeCallback()

        callbackList = [checkpoint, display_time]

        return callbackList

    print("****************************************")
    print("Getting Training Data")
    trainDatas = getTrainData.get_datas()

    print("****************************************")
    print("Getting Validation Data")
    valDatas = getValData.getDatas()

    print("****************************************")
    print("Creating Training Dataset")
    trainingDataset = createDataset.create_dataset(trainDatas).repeat()

    print("****************************************")
    print("Creating Validation Dataset")
    valDataset = createDataset.create_dataset(valDatas)

    model.fit(
        trainingDataset,
        validation_data=valDataset,
        epochs=trainEpochs,
        callbacks=getCallbacks(),
        steps_per_epoch=100,
    )
