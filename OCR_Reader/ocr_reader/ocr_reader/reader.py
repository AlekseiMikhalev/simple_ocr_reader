import click
import cv2
import pytesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pdf2image import convert_from_path

POPPLER_PATH = r'C:\Users\aleks\Downloads\poppler-21.11.0\Library\bin'


def remove_noise(cv_image):
    """
    Take the cv2-opened image and remove noise

    Parameters
    ----------
    cv_image: numpy.ndarray 
        image opened in cv2

    Returns
    -------
    converted image
    """
    kernel_1 = np.ones((1, 1), np.uint8)
    image = cv2.dilate(cv_image, kernel_1, iterations=1)
    kernel_2 = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel_2, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_2)
    noise_removed_image = cv2.medianBlur(image, 3)

    return noise_removed_image


def remove_borders(cv_image):
    """
    Take the cv2-opened image and remove borders

    Parameters
    ----------
    cv_image: numpy.ndarray 
        image opened in cv2

    Returns
    -------
    converted image
    """
    contours, hierarchy = cv2.findContours(cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cnts_sorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_image = cv_image[y:y+h, x:x+w]
    
    return cropped_image



def ocr_extraction_bounded(image_path: str) -> str:
    """
    Take image and extract text

    Parameters
    ----------
    image_path: str 

    Returns
    -------
    results: list
        extracted text
    """
    if image_path.endswith('.pdf'):
        pages = convert_from_path(image_path, poppler_path=POPPLER_PATH)
        # count to store images of each page of PDF to image
        image_counter = 1
    
        # Iterate through all the pages stored above
        for page in pages:
            # Declaring filename for each page of PDF as JPG
            # For each page, filename will be:
            # PDF page 1 -> page_1.jpg
            # PDF page 2 -> page_2.jpg
            # PDF page 3 -> page_3.jpg
            # ....
            # PDF page n -> page_n.jpg
            filename = "page_" + str(image_counter) + ".jpg"
            
            # Save the image of the page in system
            page.save(filename, 'JPEG')
        
            # Increment the counter to update filename
            image_counter = image_counter + 1

        # Variable to get count of total number of pages
        filelimit = image_counter-1
        
        # Creating a text file to write the output
        ocr_results = []
        
        # Iterate from 1 to total number of pages
        for i in range(1, filelimit + 1):
        
            # Set filename to recognize text from
            # Again, these files will be:
            # page_1.jpg
            # page_2.jpg
            # ....
            # page_n.jpg
            filename = "page_" + str(i) + ".jpg"
                
            # Recognize the text as string in image using pytesserct
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold, img_black_white = cv2.threshold(gray, 120, 250, cv2.THRESH_BINARY) # full binarization
            borders_removed = remove_borders(img_black_white)
            noise_removed = remove_noise(borders_removed) 
            extracted_page = pytesseract.image_to_string(noise_removed) 
        
            # Add extracted page into list
            ocr_results.append(extracted_page)
        ocr_results_string = ' '.join(ocr_results)
        ocr_results_string = ocr_results_string.replace('\n', ' ').replace('\xe9', ' ')
        print(ocr_results_string)

        return ocr_results_string 

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, img_black_white = cv2.threshold(gray, 120, 250, cv2.THRESH_BINARY) # full binarization
    borders_removed = remove_borders(img_black_white)
    noise_removed = remove_noise(borders_removed)
    
    ocr_results = pytesseract.image_to_string(noise_removed)

    return ocr_results


@click.command()
@click.option('-i', '--input',  help="Input file")
@click.option('-o', '--output', help="Output file")
@click.option('-v', '--verbose', is_flag=True, help="Will print verbose messages.")


def extract_and_save(input, output, verbose=True):
    """Take input file, recognize and extract text"""
    extracted_text = ocr_extraction_bounded(str(input))
    
    with open(str(output), 'w') as f:
        f.write(extracted_text)

extract_and_save()

    