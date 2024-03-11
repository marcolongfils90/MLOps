import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from ml_project.utils import common
from ml_project.pipeline import inference_pipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = inference_pipeline.InferencePipeline(self.filename)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")




@app.route("/train", methods=["GET","POST"])
@cross_origin()
def train_route():
    os.system("dvc repro")
    return "Training done successfully!"



@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    image = request.json['image']
    common.decode_image(image, clApp.filename)
    result = clApp.classifier.predict()
    return [f"The image is {result[0]['image']}."]
    # # return "{{ result[0]['image'] }}"
    # clApp.result = f"The image is {result[0]['image']}."
    # # print(clApp.result)
    # return render_template("index.html", result=clApp.result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)  # for AWS
