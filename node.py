from Flask import Flask
app = Flask(__name__)


@app.route('/initialize', methods=['POST'])
def initialize_model():
    # Receive model configuration and set epoch to 0
    # Initialize or reset local model here
    return 'Model initialized', 200
