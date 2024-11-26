import orjson
import random
import os
import concurrent.futures
import numpy as np
from pathlib import Path
from tqdm import tqdm

rootDir = Path("/tf/data/augment")

sourceDataDir = rootDir / "source"
labelDataDir = rootDir / "label"


def get_label_paths(Dir: Path):
    label_paths = [os.path.join(Dir, f) for f in os.listdir(Dir) if f.endswith(".json")]
    print("data shuffle")
    label_paths = np.random.permutation(label_paths)
    return label_paths


def get_data(label_path):
    if not Path(label_path).exists():
        return None
    try:
        with open(label_path, "r", encoding="utf-8") as file:
            data = orjson.loads(file.read())
            imageFileName = data["image"]["file_name"]
            writerNo = data["license"]["writer_no"]
        return (writerNo, sourceDataDir / imageFileName)
    except orjson.JSONDecodeError:
        # 잘못된 JSON 파일 처리
        print(f"Warning: Invalid JSON format in {label_path}")
        return None
    except KeyError as e:
        # 예상치 못한 키 오류 처리
        print(f"Warning: Missing key {e} in {label_path}")
        return None


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
            result = future.result()
            if result is not None:
                writerNo, imagePath = result
                datas.append((writerNo, imagePath))

    return datas
