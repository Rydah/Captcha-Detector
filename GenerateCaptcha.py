from captcha.image import ImageCaptcha
import random
import string
import os

folder_name = "TestCaptcha"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

letters = string.ascii_letters + string.digits
for i in range(500):
    text = ''.join(random.choice(letters) for _ in range(4))

    image = ImageCaptcha(width= 300, height= 100)
    image_data = image.generate(text)

    image_file = os.path.join(folder_name, f'{text}.png')
    with open(image_file, 'wb') as f:
        f.write(image_data.read())
