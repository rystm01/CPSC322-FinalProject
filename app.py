from flask import Flask
from flask import request
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return '<form method = "GET" action="/hello"> \
  <label for="c1_party">First Party:</label><br> \
  <input type="text" id="c1_party" name="c1_party" ><br> \
  <label for="c2_party">Second Party:</label><br> \
  <input type="text" id="c2_party" name="c2_party"><br><br> \
  \
  <label for="c1_poll">First Polling Value:</label><br> \
  <input type="text" id="c1_poll" name="c1_poll" ><br> \
  <label for="c2_party">Second Polling Value:</label><br> \
  <input type="text" id="c2_poll" name="c2_poll"><br><br> \
  <input type="submit" value="Predict"  > \
</form>'




@app.route("/hello", methods=["GET"])
def hello():
    with open('knn_model', 'rb') as file:
      knn = pickle.load(file)
    c1_party = request.args.get("c1_party")
    c2_party =request.args.get("c2_party")
    c1_polls = float(request.args.get("c1_poll"))
    c2_polls = float(request.args.get('c2_poll'))

    pred = knn.predict([[c1_polls, c2_polls, c1_party, c2_party]], categorical=True)[0]

    return "<h1>{}!</h1>".format(pred)


if __name__ == "__main__":
    app.run()


