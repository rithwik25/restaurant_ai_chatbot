from flask import Flask, render_template, request, jsonify
import time
import uuid
from zeal.backend.workflow.graph import handle_message

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    start_time = time.time()
    response = handle_message(query, session_id)
    end_time = time.time()
    
    return jsonify({
        'response': response,
        'session_id': session_id,
        'time_taken': f"{end_time - start_time:.2f}s"
    })

if __name__ == '__main__':
    app.run(debug=True)