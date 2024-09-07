from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
try:
    with open('Kidney.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract and convert form data
            sg = float(request.form['sg'])
            htn = float(request.form['htn'])
            hemo = float(request.form['hemo'])
            dm = float(request.form['dm'])
            al = float(request.form['al'])
            appet = float(request.form['appet'])
            rc = float(request.form['rc'])
            pc = float(request.form['pc'])

            # Prepare input for prediction
            values = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
            
            if model:
                prediction = model.predict(values)
                return render_template('result.html', prediction=prediction[0])
            else:
                return "Model not loaded correctly. Please check the server logs."
        except Exception as e:
            return f"Error during prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
