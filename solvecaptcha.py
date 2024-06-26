from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle


MODEL_FILENAME = "captcha_mode.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_FOLDER = "TestCaptcha"
MIN = 30

with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILENAME)

captcha_image_files = list(paths.list_images(CAPTCHA_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

for image_file in captcha_image_files:
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (5, 5), 0)

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

    output = cv2.merge([image] * 3)
    predictions = []

    for letter_bounding_box in letter_image_regions:
        x,y,w,h = letter_bounding_box

        letter_image = image[y:y+h, x:x+w]

        letter_image = resize_to_fit(letter_image, 20, 20)

        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        prediction = model.predict(letter_image)

        # convert one-hot encoded back to letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        cv2.rectangle(output, (x,y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(output, letter, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    captcha_text = "".join(predictions)
    print(f"CAPTCHA is {captcha_text}")
    print(f"FILE is {image_file}")

    cv2.imshow("Output", output)
    cv2.waitKey()