import random
import orjson
import concurrent.futures
import augment_data
import cv2
from pathlib import Path
from tqdm import tqdm


rootDir = Path("/tf/data/dataset/diff_shape_korean_data/data/Training")
subDir = Path("handWriting/char")

originalSourceDataDir = rootDir / "source" / subDir
originalLabelDataDir = rootDir / "label" / subDir

saveRootDir = Path("/tf/data/augment")
augSourceSaveDir = saveRootDir / "source"
augLabelSaveDir = saveRootDir / "label"


def get_label_paths(Dir: Path):
    label_paths = [str(path) for path in Dir.glob("*/*.json")]
    random.shuffle(label_paths)
    return label_paths


def get_data(label_path):
    with open(label_path, "r", encoding="utf-8") as file:
        data = orjson.loads(file.read())

        imageFileName = data["image"]["file_name"]
        writerNo = data["license"]["writer_no"]

    return (writerNo, originalSourceDataDir / writerNo / imageFileName)


def get_datas():
    datas = []
    label_paths = get_label_paths(originalLabelDataDir)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(get_data, label_path): label_path
            for label_path in label_paths
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_path),
            total=len(label_paths),
            desc=f"[get_datas]",
            unit="file",
        ):
            writerNo, imagePath = future.result()
            datas.append((writerNo, imagePath))

    return datas


def create_augment_file(writerNo, imagePath):
    totalCreatedCount = 0
    imageName = Path(imagePath).stem

    if Path.exists(augSourceSaveDir / f"{imageName}_0.jpg"):
        return totalCreatedCount

    augmented_list = augment_data.augment_images(image_path=imagePath)

    for idx, image in enumerate(augmented_list):
        sourceSavePath = augSourceSaveDir / f"{imageName}_{idx}.jpg"
        labelSavePath = augLabelSaveDir / f"{imageName}_{idx}.json"
        cv2.imwrite(sourceSavePath, image)

        data = {
            "image": {"file_name": sourceSavePath.name},
            "license": {"writer_no": writerNo},
        }
        with open(labelSavePath, "w", encoding="utf-8") as json_file:
            data = orjson.dumps(data).decode("utf-8")
            json_file.write(data)

        totalCreatedCount += 1

    return totalCreatedCount


totalCreatedCount = 0
datas = get_datas()
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_path = {
        executor.submit(create_augment_file, writerNo, imagePath): (
            writerNo,
            imagePath,
        )
        for (writerNo, imagePath) in datas
    }

    for future in tqdm(
        concurrent.futures.as_completed(future_to_path),
        total=len(datas),
        desc="[augment_images]",
        unit="file",
    ):
        totalCreatedCount += future.result()

print(f"totalCreatedCount: {totalCreatedCount}")
