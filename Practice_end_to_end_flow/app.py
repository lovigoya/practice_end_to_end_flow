from flask import Flask,render_template,request
from inference import *

# with open('artifacts/model.pkl', 'rb') as f:
#     model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    input_city = request.form['city']
    input_second = request.form['second']
    input_third = request.form['third']
    input_fourth = request.form['fourth']
    input_user = {"city": input_city, "Second": input_second, "Third": input_third , "Fourth": input_fourth}
    prediction = predict(input_user)
    
    # Call your function here
    return "Submitted input text: {}, {}, {}, {}, prediction: {}".format(input_city, input_second, input_third, input_fourth,prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


