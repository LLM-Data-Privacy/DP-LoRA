from flask import Flask, request, jsonify, send_file
import requests
from io import BytesIO
import torch
from ..common.prepare_dataset import prepare_dataset
from ..common.CNN_model import run_centralised
import torch
trainloaders, valloaders, testloader = prepare_dataset(num_partitions=5,batch_size=32)
LOCAL_DATASET = {
    "trainloaders": trainloaders[0],
    "valloaders": valloaders[0],
    "testloader": testloader[0]
}

app = Flask(__name__)