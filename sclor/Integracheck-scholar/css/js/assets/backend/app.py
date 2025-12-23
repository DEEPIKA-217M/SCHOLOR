from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data['text']

    return jsonify({
        "ai_probability": 72,
        "human_probability": 28,
        "citation_issue": True
    })

if __name__ == "__main__":
    app.run(debug=True)
