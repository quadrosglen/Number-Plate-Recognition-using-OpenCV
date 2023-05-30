from django.shortcuts import render
import streamlit as st
import easyocr
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from django.http import JsonResponse

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

def home(request):
    if request.method == 'POST' and request.FILES['image_file']:
        image_file = request.FILES['image_file']
        annotation_file = request.FILES['annotation_file'] if 'annotation_file' in request.FILES else None

        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if annotation_file is not None:
            number_plate = detect_number_plate(img, annotation_file)
        else:
            number_plate = detect_number_plate(img)

        if number_plate is not None:
            message = f"The number plate from the image is: {number_plate}"
        else:
            number_plate = fallback_method(img)

            if number_plate is not None:
                message = f"The number plate from the image is: {number_plate}"
            else:
                message = "Number plate could not be detected using both methods."

        context = {'message': message}
        return render(request, 'home.html', context)

    return render(request, 'home.html')

def process_image(request):
    if request.method == 'POST' and request.FILES.get('image_file'):
        image_file = request.FILES['image_file']

        # Read the uploaded image
        img = cv2.imread(image_file.temporary_file_path())

        # Perform image processing
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
                result_text = result[0][1]
                result = {
                    'result_key': result_text
                }
                return JsonResponse(result)

        return JsonResponse({'error': 'No text found'}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)
