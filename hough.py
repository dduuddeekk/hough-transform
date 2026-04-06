import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
image_file = os.path.join(script_dir, 'img.jpeg')

img = cv2.imread(image_file)

if img is None:
    print("Could not read the image.")
else:
    img_result = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(img_blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('1 - Gambar Asli', img)
    cv2.imshow('2 - Deteksi Tepi (Canny)', edges)
    cv2.imshow('3 - Hasil Hough Transform', img_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()