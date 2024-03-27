from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained classifier and vectorizer
classifier = load('classifier.joblib')
vectorizer = load('vectorizer.joblib')

# Load specialties data into a DataFrame
specialties_df = pd.read_csv('doctor_specialties.csv')  # Adjust the filename/path as necessary

# Function to recommend doctor based on symptoms
def recommend_doctor(symptoms):
    symptoms_vectorized = vectorizer.transform([symptoms])
    specialty_pred = classifier.predict(symptoms_vectorized)[0]
    
    # Find doctor for predicted specialty
    try:
        recommended_doctor = specialties_df[specialties_df['Specialty'] == specialty_pred]['Doctor_Name'].iloc[0]
    except IndexError:
        # Handle case where no doctor is found for predicted specialty
        recommended_doctor = "No doctor found"
    
    return recommended_doctor, specialty_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        recommended_doctor, specialty = recommend_doctor(symptoms)
        return render_template('index.html', symptoms=symptoms, doctor=recommended_doctor, specialty=specialty)

if __name__ == '__main__':
    app.run(debug=True)
