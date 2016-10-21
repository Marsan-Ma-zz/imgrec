# coding: utf-8
import os, json, yaml, requests, math
import numpy as np

from PIL import Image
from io import BytesIO
from datetime import datetime
from flask import Flask, request, render_template
from random import random, choice

from lib import data_util
from lib.googlenet import GoogLeNet


app = Flask(__name__)

# model
scope_name, label_size = '17flowers', 17
# scope_name, label_size = '17portraits', 9
gnet = GoogLeNet(img_size=227, label_size=label_size, gpu_memory_fraction=0.4, scope_name=scope_name)


#---------------------------
#   Server
#---------------------------
@app.route('/', methods=['GET'])
def guess():
    url = request.args.get('url', '')
    if url:
        X = url2sample(url)
        probs = gnet.predict([X])[0]
        cnt = int(sum([math.exp(i+4) * probs[i] for i in range(len(probs))]))
        probs = [(i, round(100*p, 1)) for i, p in enumerate(probs)]
    else:
        cnt, probs = None, None
    return render_template('guess.html', probs=probs, url=url, cnt=cnt)


def url2sample(url, resize=(227, 227)):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize(resize, Image.ANTIALIAS)
        img.load()
        img = np.asarray(img, dtype="float32")
        if (len(img.shape) != 3) or (img.shape[2] != 3): 
            return None
        img /= 255.
        return img
    except Exception as e:
        print(e)
        return None

#---------------------------
#   Start Server
#---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8883, debug=False)

