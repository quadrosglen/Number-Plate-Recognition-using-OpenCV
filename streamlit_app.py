import streamlit as st
import easyocr
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def detect_number_plate(img, annotation=None):
    reader = easyocr.Reader(['en'])

    if annotation is not None:
        tree = ET.parse(annotation)
        root = tree.getroot()

        xmin = int(root.find('object/bndbox/xmin').text)
        ymin = int(root.find('object/bndbox/ymin').text)
        xmax = int(root.find('object/bndbox/xmax').text)
        ymax = int(root.find('object/bndbox/ymax').text)

        cropped_image = img[ymin:ymax, xmin:xmax]

        result = reader.readtext(cropped_image)

        if len(result) > 0:
            return result[0][1]

    return None

def fallback_method(img):
    greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    enhanced_img = cv2.equalizeHist(greyscale_img)

    edges = cv2.Canny(enhanced_img, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    results_list = []

    if len(contours) > 0:
        contour = contours[0]
        (x, y, w, h) = cv2.boundingRect(contour)

        cropped_image = enhanced_img[y:y + h, x:x + w]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        if len(result) > 0:
            return result[0][1]

    return None

def main():
    st.title("Number Plate Extractor using EasyOCR")

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    annotation_file = st.file_uploader("Upload Annotation (Optional)", type=["xml"])

    if image_file is not None:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if annotation_file is not None:
            number_plate = detect_number_plate(img, annotation_file)
        else:
            number_plate = detect_number_plate(img)

        if number_plate is not None:
            st.success(f"The number plate from the image is: {number_plate}")
        else:
            number_plate = fallback_method(img)

            if number_plate is not None:
                st.success(f"The number plate from the image is: {number_plate}")
            else:
                st.error("Number plate could not be detected using both methods.")

if __name__ == "__main__":
    main()