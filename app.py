from flask import Flask, render_template, request
import pickle
import numpy as np

scaler = pickle.load(open('scaling.pkl', 'rb'))
reg_model = pickle.load(open('regmodel.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    #return 'Hello'
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_data = scaler.transform(np.array(data).reshape(1, -1))
    output = reg_model.predict(final_data)[0]
    
    return render_template('home.html', prediction_text='House Price is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)