import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



class InferencePipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model("model/trained_model.h5")

        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{"image": prediction}]

        prediction = 'Normal'
        return [{"image": prediction}]
