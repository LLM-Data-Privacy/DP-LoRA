from flask import Flask, request, send_file,jsonify
import torch
from CNN_model import run_centralised
from io import BytesIO
from aggregate import aggregate_updates, apply_updates_to_model
import os



app = Flask(__name__)
MODEL_DIR = "saved_model"
SERVER_MODEL = ServerCLIPModel.from_pretrained(MODEL_DIR)

#algorithms for sending current model to clients.
@app.route('/get-model', methods=['GET'])
def get_model():
    #Serializing models into byte streams 
    buffer = BytesIO()
    torch.save(SERVER_MODEL.model.state_dict(), buffer)    
    #Move to the start of the stream
    buffer.seek(0)

    # Send the byte stream as a response
    return send_file(buffer, mimetype='application/octet-stream', as_attachment=True, attachment_filename="model.pt")



@app.route('/update', methods=['POST'])
def update_model():
    # receiving model or parameters trained by clients
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    buffer = BytesIO(file.read())
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))    #algorithms for aggregating and updating parameters to current model
    aggregate_updates(state_dict)
    return jsonify({"message": "Model updated successfully"})

if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        SERVER_MODEL.save_pretrained(MODEL_DIR)
    app.run(debug=True, port=5000)