import orjson
import random
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

rootDir = Path("E:/augment_data/")

sourceDataDir = rootDir / "source"
labelDataDir = rootDir / "label"


def get_label_paths(Dir: Path):
    label_paths = [str(path) for path in Dir.glob("*.json")]
    random.shuffle(label_paths)
    return label_paths


def get_data(label_path):
    with open(label_path, "r", encoding="utf-8") as file:
        data = orjson.loads(file.read())

        imageFileName = data["image"]["file_name"]
        writerNo = data["license"]["writer_no"]

    return (writerNo, sourceDataDir / imageFileName)


def get_datas():
    datas = []
    label_paths = get_label_paths(labelDataDir)
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
