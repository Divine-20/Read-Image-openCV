import cv2
import pytesseract
# Tesseract path
tesseract_path = r"/usr/local/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path
img = cv2.imread("./image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# Find the single largest contour = a single ROI
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
largest_cnt = max(cnts, key=cv2.contourArea)
# Extract the ROI based on the largest contour
x, y, w, h = cv2.boundingRect(largest_cnt)
roi = img[y:y+h, x:x+w]
# Perform OCR on the ROI
text = pytesseract.image_to_string(roi, config='--psm 9 --oem 3 -c tessedit_char_whitelist=0123456789')
# Display text
print('The extracted text is ',text)