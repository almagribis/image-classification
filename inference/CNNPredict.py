import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os, sys
import argparse
import time

path_root = os.path.abspath('..')
sys.path.insert(0, path_root)

from config import settings
from utils.load_labels import DataLabels

class CNNPredict():
    def __init__(self, model_path=None):
        """load CNN pickle"""
        if model_path is None:
            model_path = os.path.join(path_root, settings.MODEL_CNN_PATH)
        self.cnn_model = load_model(model_path)
        self.data_labels = DataLabels()
    
    def preprocess_image(self, img):
        """preprocess image"""
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict_image(self, image_path:str=None, image_pil:str=None):
        start_time = time.time()
        
        if image_path:
            img = image.load_img(image_path, target_size=(28,28), color_mode="grayscale")
        else:
            img = image_pil.resize((28,28)).convert("L")
        
        img = self.preprocess_image(img=img)
        
        """run prediction"""
        prediction = self.cnn_model.predict(img)

        """get probabilit score"""
        predicted_class = np.argmax(prediction)
        prediction_label = self.data_labels.get_labels(predicted_class)
        probabilities = np.max(prediction)

        inference_time = time.time() - start_time
        return {"label":prediction_label, "score":round(float(probabilities), 3), "inference_time":round(float(inference_time), 3)}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="image path")
    args = parser.parse_args()
    svc_predict = CNNPredict()
    result = svc_predict.predict_image(image_path=args.image_path)
    print(result)