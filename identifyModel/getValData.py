import orjson
import random
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
con = os.environ.get("con")

rootDir = (
    Path("/tf/data/dataset/diff_shape_korean_data/data/Validation")
    if con == "0"
    else Path("e:/diff_shape_korean_data/data/Validation")
)
subDir = Path("handWriting/char")

sourceDataDir = rootDir / "source" / subDir
labelDataDir = rootDir / "label" / subDir


def getLabelPaths(Dir: Path):
    labelPaths = [str(path) for path in Dir.glob("*/*.json")]
    random.shuffle(labelPaths)
    return labelPaths


def getData(labelPath):
    with open(labelPath, "r", encoding="utf-8") as file:
        data = orjson.loads(file.read())

        imageFileName = data["image"]["file_name"]
        writerNo = data["license"]["writer_no"]

    return (writerNo, sourceDataDir / writerNo / imageFileName)


def getDatas():
    datas = []
    labelPaths = getLabelPaths(labelDataDir)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futureToPath = {
            executor.submit(getData, labelPath): labelPath for labelPath in labelPaths
        }

        for future in tqdm(
            concurrent.futures.as_completed(futureToPath),
            total=len(labelPaths),
            desc="[getDatas]",
            unit="file",
        ):
            writerNo, imagePath = future.result()
            datas.append((writerNo, imagePath))
    return datas
