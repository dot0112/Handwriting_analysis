import cv2
import numpy as np
import random
import os


def augment_image(image, target_size=(110, 110)):
    """
    이미지 증강 함수
    기울기, 크기, 위치, 공백 처리를 수행합니다.
    """
    # 1. 기울기: 최대 15도
    angle = random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (w, h), borderValue=(255, 255, 255)
    )  # 흰색 배경

    # 2. 크기: 85% ~ 115%
    scale_factor = random.uniform(0.8, 0.95)
    new_width = int(rotated_image.shape[1] * scale_factor)
    new_height = int(rotated_image.shape[0] * scale_factor)
    resized_image = cv2.resize(rotated_image, (new_width, new_height))

    # 3. 위치: 중앙에서 ±25% 이내로 이동
    max_shift_x = int(resized_image.shape[1] * 0.08)
    max_shift_y = int(resized_image.shape[0] * 0.08)
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


def display_image(image):
    """
    증강된 이미지를 화면에 띄우는 함수
    이미지 크기를 화면 크기에 맞추도록 자동 조정
    """
    max_width = 300
    max_height = 300

    height, width = image.shape[:2]
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))

    cv2.imshow("Augmented Image", resized_image)
    cv2.resizeWindow("Augmented Image", new_width, new_height)

    # 사용자가 키 입력을 기다립니다 (여기서 'b' 키를 누르면 이미지 저장)
    key = cv2.waitKey(0) & 0xFF
    return key


def get_unique_filename(base_path):
    """
    주어진 경로에 파일이 존재하는지 확인하고, 존재하면 이름을 변경하여 반환합니다.
    예) gage.jpg -> gage1.jpg -> gage2.jpg 형식으로 변경
    """
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    counter = 1
    new_path = f"{base}{counter}{ext}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}{counter}{ext}"

    return new_path


def process_image(image_path, save_path):
    """
    이미지를 110x110으로 리사이즈하고 증강한 후 결과를 화면에 띄우며, 저장하는 함수
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 원본 이미지를 110x110으로 리사이즈
    image_resized = cv2.resize(image, (110, 110))

    # 이미지 증강 처리
    augmented_image = augment_image(image_resized)

    # 화면에 이미지 표시
    key = display_image(augmented_image)

    # 이미지 처리 완료 메시지 출력
    if key == ord("e"):  # 'b' 키가 눌리면 저장
        print("이미지 처리 완료")
        save_path = get_unique_filename(save_path)  # 고유한 파일 이름 생성
        cv2.imwrite(save_path, augmented_image)
        print(f"이미지가 저장되었습니다: {save_path}")
    elif key == ord("q"):  # 'q' 키가 눌리면 종료
        print("프로그램을 종료합니다.")
        return False  # 종료 조건
    return True  # 계속 진행


if __name__ == "__main__":
    image_path = "god.jpg"  # 실제 이미지 경로를 입력하세요.
    save_path = "god.jpg"  # 저장할 경로와 파일 이름

    print(
        "이미지 처리 프로그램이 시작되었습니다. 'b' 키를 눌러 이미지를 처리하고 저장하세요."
    )

    while True:
        continue_processing = process_image(image_path, save_path)
        if not continue_processing:
            break

    cv2.destroyAllWindows()
