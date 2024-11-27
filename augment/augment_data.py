import random
import cv2
import numpy as np


def augment_image(image, target_size=(110, 110)):
    """
    이미지 증강 함수
    기울기, 크기, 위치, 공백 처리를 수행합니다.
    """
    # 1. 기울기: 최대 15도
    angle = random.uniform(-20, 20)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (w, h), borderValue=(255, 255, 255)
    )  # 흰색 배경

    # 2. 크기: 85% ~ 115%
    scale_factor = random.uniform(0.75, 0.95)
    new_width = int(rotated_image.shape[1] * scale_factor)
    new_height = int(rotated_image.shape[0] * scale_factor)
    resized_image = cv2.resize(rotated_image, (new_width, new_height))

    # 3. 위치: 중앙에서 ±25% 이내로 이동
    max_shift_x = int(resized_image.shape[1] * 0.12)
    max_shift_y = int(resized_image.shape[0] * 0.12)
    shift_x = random.randint(-max_shift_x, max_shift_x)
    shift_y = random.randint(-max_shift_y, max_shift_y)

    # 4. 최종 이미지 크기(target_size)로 맞추기 위해 이동 후 공백을 흰색으로 채운다
    height, width = target_size
    result_image = np.full((height, width, 3), 255, dtype=np.uint8)  # 흰색 배경

    # 자를 범위 계산 (이동한 후 이미지 크기)
    start_x = max(0, shift_x)
    start_y = max(0, shift_y)
    end_x = min(resized_image.shape[1], shift_x + width)
    end_y = min(resized_image.shape[0], shift_y + height)

    # result_image의 시작 위치를 맞추기 위해서 end_x, end_y가 target_size를 넘지 않도록 처리
    insert_start_x = max(0, -shift_x)
    insert_start_y = max(0, -shift_y)
    insert_end_x = insert_start_x + (end_x - start_x)
    insert_end_y = insert_start_y + (end_y - start_y)

    # 이미지를 적절한 위치에 삽입
    result_image[insert_start_y:insert_end_y, insert_start_x:insert_end_x] = (
        resized_image[start_y:end_y, start_x:end_x]
    )

    # 5. 최종 크기 조정
    final_image = cv2.resize(result_image, target_size)

    return final_image


def augment_images(image_path) -> list:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    augmented_list = [image]
    for _ in range(5):
        # 원본 이미지를 110x110으로 리사이즈
        image_resized = cv2.resize(image, (110, 110))
        augmented_image = augment_image(image_resized)
        augmented_list.append(augmented_image)

    return augmented_list
