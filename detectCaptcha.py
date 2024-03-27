import cv2
import glob
import os
import imutils

CAPTCHA_IMAGE_FOLDER = 'CaptchaImages'
OUTPUT_FOLDER = "ExtractedLetter"
MIN = 100

captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

for (i, captcha_image_file) in enumerate(captcha_image_files):
    print(f"[INFO] Processing image {i+1}/{len(captcha_image_files)}")

    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image (convert it to pure black and white)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find the contours (continuous blobs of pixels) the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    letter_image_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN:
            (x,y,w,h) = cv2.boundingRect(contour)

            if w/h > 1.25:
                half_width = int(w/2)
                letter_image_regions.append((x,y,half_width,h))
                letter_image_regions.append((x+half_width,y,half_width,h))
            else:
                letter_image_regions.append((x,y,w,h))


    if len(letter_image_regions) != 4:
        continue

    letter_image_regions = sorted(letter_image_regions, key=lambda x:x[0])

    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = gray[y:y + h, x:x + w]

        print(letter_text)
        # Get the folder to save the image in
        if letter_text.isupper():
            save_path = os.path.join(OUTPUT_FOLDER, "uppercase", letter_text)
        elif letter_text.islower():
            save_path = os.path.join(OUTPUT_FOLDER, "lowercase", letter_text)
        else:
            save_path = os.path.join(OUTPUT_FOLDER, "numbers", letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1

print(counts)
