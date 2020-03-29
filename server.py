import io
from flask import Flask, render_template, request, Response
import scaling_demo

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scaling")
def scaling():
    return render_template(
        "scaling.html",
        scalers=scaling_demo.get_scaler_names(),
        datasets=scaling_demo.get_dataset_names(),
    )


@app.route("/scaling-viz")
def scaling_viz():
    dataset = request.args.get("dataset")
    scaler = request.args.get("scaler")
    fig, axs = scaling_demo.visualize_scaler(scaler, dataset)
    output = io.BytesIO()
    fig.savefig(output, format="png")
    return Response(output.getvalue(), mimetype="image/png")


if __name__ == "__main__":
    import waitress
    import sys

    waitress.seve(app, port=5000 if len(sys.argv) < 2 else int(sys.argv[1]))
