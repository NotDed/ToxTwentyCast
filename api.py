from flask import Flask
from flask import request, jsonify
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

save = "40eRun.bin"
model = torch.load(save).cuda()

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def query():
    resultados = {}
    if request.method == 'POST':
        data = dict(request.json)
        print('aaaaaaaaa')
        print(data)
        resultados = multiPredict(model, tokenizer, data['selfie'])
        return resultados
        
if __name__ == "__main__":
    app.run(port = 3000, debug = True)