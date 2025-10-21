from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from flask import send_file
import pandas as pd
import io

app = Flask(__name__)
# Load trained model
model = joblib.load("model_trainer/certificate_model.pkl")
print("ML model loaded successfully!")

# ---- Load Model ----
def mock_model_predict(data):
    return model.predict(data)

# ---- Web UI ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_certificate():
    try:
        # Simulate data input (later we’ll parse actual form fields)
        data = request.form.getlist('scores', type=float)
        result = mock_model_predict(data)
        return render_template('result.html', status=result)
    except Exception as e:
        return render_template('result.html', status=f"Error: {e}")

# ---- API Endpoint ----
@app.route('/api/submit', methods=['POST'])
def api_submit():
    data = request.get_json()

    if not data or "data" not in data:
        return jsonify({"error": "Missing 'data' in request"}), 400

    features = np.array(data["data"]).reshape(1, -1)

    try:
        prediction = model.predict(features)[0]
        result = "Pass" if prediction == 1 else "Fail"
        return jsonify({
            "status": "success",
            "prediction": result,
            "message": f"Certificate processed: {result}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read CSV
        df = pd.read_csv(file)
        if df.shape[1] != 11:
            return jsonify({"error": "CSV must have exactly 11 columns"}), 400

        # Make predictions
        predictions = model.predict(df.values)
        results = ["Pass" if p == 1 else "Fail" for p in predictions]
        df['Prediction'] = results

        # Save updated CSV to memory
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='certificate_results.csv'
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/docs')
def api_docs():
    return render_template('docs.html')

if __name__ == '__main__':
    app.run(debug=True)
