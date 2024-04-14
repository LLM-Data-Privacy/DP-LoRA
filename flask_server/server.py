from flask import Flask, request, jsonify
from threading import Lock

app = Flask(__name__)
clients_data = {}
client_count = 3
lock = Lock()

@app.route("/broadcast", methods=["GET"])
def broadcast():
    initial_int = 20
    return jsonify({"number": initial_int}), 200

@app.route("/collect", methods=["POST"])
def collect_date():
    global clients_data
    data = request.get_json
    client_id = data['client_id']
    number = data['number']
    
    with lock:
        clients_data[client_id] = number
        if len(clients_data) == client_count: # all clients have sent their data
            average_sum = sum(clients_data.values()) / client_count
            clients_data = {}
            
            return jsonify({"sum": sum}), 200
        
    return jsonify({"message": "Data received"}), 200

if __name__ == "__main__":
    app.run(debug=True,port=8000)