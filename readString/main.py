import sys
import os
from PIL import Image

argv = sys.argv
scriptFileName = os.path.basename(argv[0])
currDir = os.path.dirname(os.path.realpath(__file__))

print()
print("Usage:")
print("     python %s <Image Path>" % scriptFileName)
print()

if len(argv) < 2:
    raise ValueError("No image path specified")

imagePath = argv[1]

print("     Image Path: %s" % imagePath)

try:
    image = Image.open(imagePath)
except:
    raise ValueError("Invalid Image Path: %s" % imagePath)

imageName = os.path.splitext(os.path.basename(imagePath))[0]
imageSavePath = os.path.join(currDir, "Result", imageName, "")
os.makedirs(os.path.dirname(imageSavePath), exist_ok=True)

import easyocr

reader = easyocr.Reader(["ko", "en"])

# string
recoglist = []

result = reader.readtext(imagePath)
for coords, value, confi in result:
    box_coord = (coords[0][0], coords[0][1], coords[1][0], coords[2][1])
    cropImage = image.crop(box_coord)
    cropImage.save(os.path.join(imageSavePath, "string_" + value + ".jpg"), "JPEG")
    recoglist.extend(value)


# char
i = 0
result = reader.readtext(imagePath, link_threshold=1.0, allowlist=recoglist)
for coords, value, confi in result:
    box_coord = (coords[0][0], coords[0][1], coords[1][0], coords[2][1])
    cropImage = image.crop(box_coord)
    cropImage.save(os.path.join(imageSavePath, f"char_{value}_{i}.jpg"), "JPEG")
    i += 1
