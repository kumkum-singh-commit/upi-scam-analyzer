from flask import Flask, request, jsonify, render_template
from PIL import Image
from analyzer import analyze, decode_qr

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def run_analysis():
    text = (request.form.get("text") or "").strip()
    payload = None
    qr_error = None

    f = request.files.get("qr")
    if f and f.filename:
        try:
            payload = decode_qr(Image.open(f.stream))
        except Exception:
            payload = None
        if not payload:
            qr_error = "Could not read that QR code. Try a clearer, cropped image."

    if not text and not payload:
        return jsonify(error=qr_error or "Paste a message or upload a QR code first."), 400

    score, label, reasons = analyze(text, payload)
    return jsonify(score=score, label=label, reasons=reasons,
                   qr_payload=payload, qr_error=qr_error)


if __name__ == "__main__":
    app.run(debug=False)