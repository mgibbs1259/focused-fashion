from flask import Flask, render_template, request
#from flask_dropzone import Dropzone


app = Flask(__name__)
#dropzone = Dropzone(app)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run()

print('test')