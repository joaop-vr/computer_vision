import cv2
import numpy as np
import sys
import os

if len(sys.argv) != 3:
    print("Use: python3 main.py <image> <output_path>")
    exit(1)

image_path = sys.argv[1]
output_path = sys.argv[2]
output_path += "/" if not output_path.endswith("/") else ""

os.makedirs(output_path, exist_ok=True)

points = []

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Imagem", img)

img = cv2.imread(image_path)
clone = img.copy()

cv2.namedWindow("Imagem", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Imagem", 800, 600)

cv2.imshow("Imagem", img)
cv2.setMouseCallback("Imagem", mouse_handler)
cv2.waitKey(0)

w, h = 600, 800
dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
src = np.array(points, dtype='float32')

M = cv2.getPerspectiveTransform(src, dst)
warp = cv2.warpPerspective(clone, M, (w, h))

cv2.imshow("Imagem", warp)
cv2.waitKey(0)
cv2.imwrite(f'{output_path}{image_path.split("/")[-1].split(".")[0]}.jpg', warp)
cv2.destroyAllWindows()
