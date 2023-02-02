from deepface import DeepFace
import base64
from io import BytesIO

from PIL import Image
import numpy as np
import sys

im_bytes = base64.b64decode(sys.argv[1])

im_bytes2 = base64.b64decode(sys.argv[2])
im_file = BytesIO(im_bytes)
im_file2 = BytesIO(im_bytes2)
img = Image.open(im_file)   # img is now PIL Image object
img2 = Image.open(im_file2)   # img is now PIL Image object
tt = np.array(img)     ## # convert image to file-like object

tt2 = np.array(img2) 








result = DeepFace.verify(tt,tt2,detector_backend='dlib', model_name = 'Dlib')

print(result)
print(result["distance"])
faceMatchingScore="{:.8f}".format(float(result["distance"]))
print(faceMatchingScore)
