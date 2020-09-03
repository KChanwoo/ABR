"""
@Author Chanwoo Kwon, Yonsei Univ. Researcher since 2020.05~
"""
from flask import Flask, jsonify, request
from network.RNN import Net
from imagelib.extractor import Extractor
from config import configure

import os

app = Flask(__name__)

net = Net('./data/Response-result-origin.txt')
extractor = Extractor()
version = "0.1"
key = "maistabr"
iv = "initvector"


@app.route("/")
def hello():
    return "Hello World"


@app.route("/abr/image/predict", methods=['POST'])
def upload():
    file = request.files['file']
    value = request.values

    id = value["id"]

    if file:
        try:
            fpath = os.path.join("./", file.filename)
            file.save(fpath)

            vector = extractor.extract(fpath, True)
            tensor = net.vector_to_data(vector)
            predict = net.predict(200, tensor)

            pred = net.to_top_predict(predict)

            result = []
            for i in range(len(vector)):
                result.append({
                    "graph": vector[i],
                    "peak": pred[i]
                })

            return jsonify({
                "id": id,
                "version": version,
                "result": "success",
                "data": {
                    "extract": result
                }
            })
        except Exception as e:
            return jsonify({
                "id": id,
                "version": version,
                "result": "fail",
                "message": str(e)
            })
    else:
        return jsonify({
            "id": id,
            "version": version,
            "result": "fail",
            "message": "No file"
        })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=configure["port"])
