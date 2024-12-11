from flask import Flask
from flask import request
import pickle



app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    return ' <form action="/predict" method="get" > \
        <label for="c1_party">First Party:</label> <br>\
        <select id="c1_party" name="c1_party"> \
            <option value="DEMOCRAT">Democrat</option> \
            <option value="REPUBLICAN">Republican</option> \
        </select> <br>\
        \
        <label for="c1_poll">First Poll Result:</label>  <br>\
        <input type="text" id="c1_poll" name="c1_poll"> <br> <br>\
        \
        <label for="c2_party">Second Party:</label>  <br>\
        <select id="c2_party" name="c2_party"> \
            <option value="DEMOCRAT">Democrat</option> \
            <option value="REPUBLICAN">Republican</option> \
        </select> <br>\
         \
        <label for="c2_poll">Second Poll Result:</label> <br>\
        <input type="text" id="c2_poll" name="c2_poll"> <br><br>\
        \
        <input type="submit" value="Predict"> \
        </form>'




@app.route('/predict', methods=['GET'])
def predict():
    c1_party = request.args.get('c1_party')
    c2_party = request.args.get('c2_party')
    c1_pct = float(request.args.get('c1_poll'))
    c2_pct = float(request.args.get('c2_poll'))

    print([c1_pct, c2_pct, c1_party, c2_party])

    with open('knn_model', 'rb') as file:
        knn = pickle.load(file)
    
    pred = knn.predict([[c1_pct, c2_pct, c1_party, c2_party]], categorical=True)[0]

    return "<h1> {}!</h1>".format(pred)
    
if __name__ == "__main__":
    app.run()


