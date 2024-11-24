import easyocr
from PIL import Image, ImageDraw

imgPath = "./readString/13940002021.jpg"

# EasyOCR 리더 객체 생성 (한국어와 영어 지원)
reader = easyocr.Reader(["ko", "en"])

# 이미지에서 텍스트 읽기
result = reader.detect(imgPath, link_threshold=1.0)
result = result[0][0]
print(result)

img = Image.open(imgPath)
draw = ImageDraw.ImageDraw(img)

for coords in enumerate(result):
    box_coords = (coords[0], coords[2], coords[1], coords[3])
    draw.rectangle(box_coords, outline=(255, 0, 0), width=2)

img.show()
