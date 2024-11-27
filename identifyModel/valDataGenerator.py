import random
import orjson
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
con = os.environ.get("con")


def add_padding(image: Image.Image, new_size=110, color=(255, 255, 255)):
    width, height = image.size

    padding_width = (new_size - width) // 2
    padding_height = (new_size - height) // 2

    result = Image.new(image.mode, (new_size, new_size), color)

    result.paste(image, (padding_width, padding_height))

    return result


rootDir = (
    Path("/tf/data/dataset/diff_shape_korean_data/data/")
    if con == "0"
    else Path("e:/diff_shape_korean_data/data/")
)

valLabelDir = rootDir / "Validation" / "label" / "handWriting" / "char"
valSourceDir = rootDir / "Validation" / "source" / "handWriting" / "char"

wordLabelDir = rootDir / "Training" / "label" / "handWriting" / "word"
wordSourceDir = rootDir / "Training" / "source" / "handWriting" / "word"

# 각 인덱스 주소에서 랜덤한 100개의 데이터 선택 후 val 데이터 생성
i = 0

for idx in tqdm(range(1, 153), desc="전체 진행률", position=0, leave=True):
    idxStr = f"{idx:03d}"
    labelPaths = list(Path(wordLabelDir / idxStr).glob("*.json"))
    random_sample = random.sample(labelPaths, min(len(labelPaths), 300))

    (valSourceDir / idxStr).mkdir(parents=True, exist_ok=True)
    (valLabelDir / idxStr).mkdir(parents=True, exist_ok=True)

    # Nested progress bar for each folder
    for labelPath in tqdm(
        random_sample, desc=f"폴더 {idxStr} 처리 중", position=1, leave=False
    ):
        with open(labelPath, "r", encoding="utf-8") as file:
            data = orjson.loads(file.read())

            words = data["text"]["word"]
            fileName = data["image"]["file_name"]
            writerNo = data["license"]["writer_no"]

        imagePath = wordSourceDir / idxStr / fileName

        image = Image.open(imagePath)
        for word in words:
            x1, y1, x2, y2 = word["charbox"]
            cropImageName = f"char_{idxStr}_{i}.jpg"
            labelName = f"char_{idxStr}_{i}.json"
            i += 1
            cropImage = image.crop((x1, y1, x2, y2))
            cropImage = add_padding(cropImage)
            cropImage.save(valSourceDir / writerNo / cropImageName)
            data = {
                "image": {"file_name": cropImageName},
                "license": {"writer_no": writerNo},
            }

            with open((valLabelDir / writerNo / labelName), "w") as json_file:
                data = orjson.dumps(data).decode("utf-8")
                json_file.write(data)
