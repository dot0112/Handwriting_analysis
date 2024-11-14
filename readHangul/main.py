import sys
import os
import shutil

argv = sys.argv
scriptFileName = os.path.basename(argv[0])
currDir = os.path.dirname(os.path.realpath(__file__))

print()
print("Usage:")
print("    python %s train [<timeInHours>]" % scriptFileName)
print("    python %s val" % scriptFileName)
print()

trainHours = 65535
if argv[2] == "train":
    isTrain = True
    if len(argv) >= 4:
        trainHours = float(argv[3])

elif argv[2] == "val":
    isTrain = False


import time
import keras

modelPath = os.path.join(currDir, "readHangul" + ".model.h5")
backupModelPath = os.path.join(currDir, "readHangul" + ".model.backup.h5")


def formatTime(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


class Perf:
    def __init__(self, text, length):
        self.text = text
        self.length = length
        self.startTime = time.perf_counter()
        print("%s: 0 / %g" % (text, length), end="")

    def progress(self, value, text=None):
        print("\r%s: %g / %g, " % (self.text, value, self.length), end="")

        if text is not None:
            print("%s, " % text, end="")

        etr = (
            "nan"
            if value <= 0
            else formatTime(
                ((time.perf_counter() - self.startTime) / value * (self.length - value))
            )
        )
        print("ETR = %s" % etr, end="")

    def finish(self, text=None):
        print("\r%s completed, " % self.text, end="")

        if text is not None:
            print("%s, " % text, end="")

        elapsed = formatTime(time.perf_counter() - self.startTime)
        print("Elapsed = %s" % elapsed)


if isTrain:
    model = None
    if os.path.isfile(modelPath):
        print("Loading existing model")
        loadedModel = keras.models.load_model(modelPath)
        model.set_weights(loadedModel.get_weights())
    else:
        print("Creating new model")
        # 모델 생성 구문 필요

    print("***********************************")
    print("Start training for %g hours" % trainHours)
    # 파라미터 설정 구문 필요
    endTime = time.perf_counter() + trainHours * 3600
    epochCount = 0
    stageCount = 0
