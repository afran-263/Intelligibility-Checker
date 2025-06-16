from flask import Flask, request,jsonify, render_template
from asr1 import transcribe_audio,wer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe_and_wer', methods=['POST'])
def transcribe_and_wer():
    data = request.get_json()
    reference = data.get('reference', '')
    try:
        text = transcribe_audio()
        wer_result = wer(reference, text)
        return jsonify({
            'predicted_text': text,
            'wer_result': wer_result
        })
    except Exception as e:
        print("ASR backend error:", e)
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
