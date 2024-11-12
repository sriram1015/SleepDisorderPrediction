from flask import Flask, request, render_template
import numpy as np
import joblib

# Load your trained model (replace 'model.pkl' with your actual model file)
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            gender = int(request.form['gender'])
            age = int(request.form['age'])
            occupation = int(request.form['occupation'])
            sleep_duration = float(request.form['sleep_duration'])
            quality_of_sleep = int(request.form['quality_of_sleep'])
            physical_activity_level = int(request.form['physical_activity_level'])
            stress_level = int(request.form['stress_level'])
            bmi_category = int(request.form['bmi_category'])
            heart_rate = int(request.form['heart_rate'])
            daily_steps = int(request.form['daily_steps'])
            blood_pressure_upper = int(request.form['blood_pressure_upper'])
            blood_pressure_lower = int(request.form['blood_pressure_lower'])

            # Map integer values to their readable labels
            gender_label = "Male" if gender == 1 else "Female"
            occupation_label_map = {
                1: "Software Engineer",
                2: "Doctor",
                3: "Sales Representative",
                4: "Teacher",
                5: "Nurse"
            }
            occupation_label = occupation_label_map.get(occupation, "Unknown Occupation")
            bmi_label_map = {
                1: "Underweight",
                2: "Normal Weight",
                3: "Overweight",
                4: "Obese"
            }
            bmi_label = bmi_label_map.get(bmi_category, "Unknown BMI Category")

            # Prepare the input features for the model
            input_features = np.array([[gender, age, occupation, sleep_duration, quality_of_sleep,
                                        physical_activity_level, stress_level, bmi_category, heart_rate, 
                                        daily_steps, blood_pressure_upper, blood_pressure_lower]])

            # Make prediction
            prediction = model.predict(input_features)[0]
            
            # Map the model output to readable labels
            prediction_map = {
                0: "No Disorder",
                1: "Sleep Apnea",
                2: "Insomnia",
                3: "Narcolepsy"
            }
            
            # Convert numeric prediction to a readable label
            prediction_label = prediction_map.get(prediction, "Unknown Disorder")

            # Render the result on the same page with input details
            details = {
                'gender': gender_label,
                'age': age,
                'occupation': occupation_label,
                'sleep_duration': sleep_duration,
                'quality_of_sleep': quality_of_sleep,
                'physical_activity_level': physical_activity_level,
                'stress_level': stress_level,
                'bmi_category': bmi_label,
                'heart_rate': heart_rate,
                'daily_steps': daily_steps,
                'blood_pressure_upper': blood_pressure_upper,
                'blood_pressure_lower': blood_pressure_lower
            }

            return render_template('index.html', prediction=prediction_label, details=details)
        
        except KeyError as e:
            return f"Missing key: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)
