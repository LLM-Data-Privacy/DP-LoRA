from flask import Flask, request, jsonify
from threading import Lock

app = Flask(__name__)
clients_data = {}
client_count = 1
lock = Lock()
base_data:float = 20.0

@app.route("/broadcast", methods=["GET"])
def broadcast():
    return jsonify({"number": base_data}), 200

@app.route("/collect", methods=["POST"])
def collect_data():
    global clients_data, base_data, client_count
    data = request.get_json()
    client_id = data['client_id']
    number = data['number']
    
    with lock:
        clients_data[client_id] = number
        if len(clients_data) == client_count: # all clients have sent their data
            base_data = sum(clients_data.values()) / client_count
            clients_data = {}
            print("Sum of numbers: ", base_data)
            return jsonify({"ready": "aggregation complete"}), 200
    
    print("still need to wait for other clients to upload data")        
    return jsonify({"wait": client_count - len(clients_data.keys())}), 200

if __name__ == "__main__":
    app.run(debug=True,port=3000) 