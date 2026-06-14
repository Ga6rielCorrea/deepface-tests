import requests
import numpy as np
import cv2
from deepface import DeepFace
from google.colab.patches import cv2_imshow

def load_image_from_url(url):
    resp = requests.get(url, stream=True).raw
    image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

url = "https://admin.cnnbrasil.com.br/wp-content/uploads/sites/12/2025/01/Neymar-Comemoracao-Santos-e1738884580565.jpg"
img1 = load_image_from_url(url)


result = DeepFace.analyze(img1)
print(result)

cv2_imshow(img1)
