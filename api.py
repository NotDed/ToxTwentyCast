from flask import Flask
from flask import request, jsonify, render_template
from flask_cors import CORS

from operator import index
import pandas as pd
import sys
import torch
import json

from transformers import AutoTokenizer
from modelFunctions import predict, multiPredict

seyonecModel = 'seyonec/BPE_SELFIES_PubChem_shard00_160k'
tokenizer = AutoTokenizer.from_pretrained(seyonecModel, padding=True)

save = "100_linear_tox_modof.bin"
model = torch.load(save).cuda()

app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/')
def page():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])
def query():
    resultados = {}
    if request.method == 'POST':
        data = dict(request.json)
        resultados = multiPredict(model, tokenizer, data['selfies'])
        return resultados
        
if __name__ == "__main__":
    app.run(port = 3000, debug = True)