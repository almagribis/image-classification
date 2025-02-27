import cv2
import numpy as np
import joblib
import os, sys
import argparse
import time

path_root = os.path.abspath('..')
sys.path.insert(0, path_root)

from config import settings
from utils.load_labels import DataLabels

class SVCPredict():
    def __init__(self, model_path=None):
        """load SVC pickle"""
        if model_path is None:
            model_path = os.path.join(path_root, settings.MODEL_SVC_PATH)
        self.svc_model = joblib.load(model_path)
        self.data_labels = DataLabels()
    
    def preprocess_image(self, img):
        """preprocess image"""
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.flatten().reshape(1, -1)
        return img

    def predict_image(self, image_path:str=None, image_pil:str=None):
        start_time = time.time()
        
        if image_path:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.array(image_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = self.preprocess_image(img=img)
        
        """run prediction"""
        prediction = self.svc_model.predict(img)[0]

        """get probabilit score"""
        prediction_label = self.data_labels.get_labels(prediction)
        probabilities = np.max(self.svc_model.predict_proba(img)[0])

        inference_time = time.time() - start_time
        return {"label":prediction_label, "score":round(float(probabilities), 3), "inference_time":round(float(inference_time), 3)}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="image path")
    args = parser.parse_args()
    svc_predict = SVCPredict()
    result = svc_predict.predict_image(image_path=args.image_path)
    print(result)