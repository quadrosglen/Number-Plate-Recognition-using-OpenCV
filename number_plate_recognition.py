# -*- coding: utf-8 -*-
"""npr_single_file.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RQoo08cAnoHFbuWor7CKIP7lOARoYrAH
"""

!pip install easyocr

import easyocr
import cv2

img = cv2.imread('/content/Cars48.png')

greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

reader = easyocr.Reader(['en'])

enhanced_img = cv2.equalizeHist(greyscale_img)

enhanced_img = cv2.equalizeHist(greyscale_img)

edges = cv2.Canny(enhanced_img, 30, 100)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

results_list = []

if len(contours) > 0:
    contour = contours[0]
    (x, y, w, h) = cv2.boundingRect(contour)

    cropped_image = enhanced_img[y:y + h, x:x + w]

    # OCR using Tesseract or Google Cloud Vision OCR
    result = reader.readtext(cropped_image)

    if len(result) > 0:
        results_list.append([img, result[0][1]])

for result in results_list:
    print(f"The number plate from the image is: {result[1]}")

