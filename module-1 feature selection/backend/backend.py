from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os

from xgboost3 import run_pipeline

app = Flask(__name__)
CORS(app)

@app.route("/run-model", methods=["POST"])
def run_model():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Windows locks open file handles — must close before passing path to pipeline
    suffix = os.path.splitext(file.filename)[1] or ".csv"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp.name
    try:
        file.save(temp_path)
        temp.close()                        # close handle so Windows releases the lock
        results = run_pipeline(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(temp_path)            # safe to delete now — handle is closed
        except OSError:
            pass                            # already gone, ignore

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)