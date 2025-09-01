# --- Step 1: Import Necessary Libraries ---
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# --- Step 2: Load and Train the Machine Learning Model ---

# URL for the public Cleveland Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Define the column names as per the dataset's documentation
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load the data using pandas, marking '?' as a missing value
df = pd.read_csv(url, header=None, names=columns, na_values='?')

# Preprocessing: For simplicity, we drop rows with any missing values
df.dropna(inplace=True)

# The 'target' column has values > 0 for disease presence. We'll make this a binary (0 or 1) problem.
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Define our features (X) and the variable we want to predict (y)
X = df.drop('target', axis=1)
y = df['target']

# Initialize the Random Forest Classifier model
# We train it on the *entire* dataset for the final application
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


# --- Step 3: Define the Prediction Function for Gradio ---

# This function will take inputs from the UI, process them, and return a prediction
def predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Convert user-friendly text inputs from the UI to the numerical format the model needs
    sex_numeric = 1 if sex == "Male" else 0
    exang_numeric = 1 if exang == "Yes" else 0

    # Create a pandas DataFrame from the user's inputs
    # The column order and names must exactly match the ones used for training
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_numeric],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang_numeric],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Use the trained model to make a prediction on the input data
    prediction = model.predict(input_data)[0]

    # Return a formatted, user-friendly string based on the prediction
    if prediction == 1:
        return "🔴 **Prediction: High Chance of Heart Disease**"
    else:
        return "🟢 **Prediction: Low Chance of Heart Disease**"


# --- Step 4: Build the Gradio User Interface ---

# Define the list of input components for the Gradio interface
# Using 'value' instead of 'default' for compatibility with different Gradio versions
inputs = [
    gr.Slider(minimum=20, maximum=80, step=1, value=52, label="Age"),
    gr.Radio(choices=["Male", "Female"], value="Male", label="Sex"),
    gr.Dropdown(
        label="Chest Pain Type (cp)",
        choices=[("Typical Angina", 0), ("Atypical Angina", 1), ("Non-Anginal Pain", 2), ("Asymptomatic", 3)],
        value=2
    ),
    gr.Slider(minimum=90, maximum=200, step=1, value=128, label="Resting Blood Pressure (trestbps)"),
    gr.Slider(minimum=100, maximum=600, step=1, value=204, label="Serum Cholesterol (chol) in mg/dl"),
    gr.Radio(label="Fasting Blood Sugar > 120 mg/dl (fbs)", choices=[("False", 0), ("True", 1)], value=0),
    gr.Dropdown(
        label="Resting ECG Results (restecg)",
        choices=[("Normal", 0), ("ST-T wave abnormality", 1), ("Left Ventricular Hypertrophy", 2)],
        value=1
    ),
    gr.Slider(minimum=60, maximum=220, step=1, value=156, label="Maximum Heart Rate Achieved (thalach)"),
    gr.Radio(choices=["No", "Yes"], value="No", label="Exercise Induced Angina (exang)"),
    gr.Slider(minimum=0.0, maximum=6.2, step=0.1, value=1.0, label="ST Depression (oldpeak)"),
    gr.Dropdown(
        label="Slope of Peak Exercise ST Segment (slope)",
        choices=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
        value=2
    ),
    gr.Dropdown(label="Major Vessels Colored by Flourosopy (ca)", choices=[0, 1, 2, 3], value=0),
    gr.Dropdown(
        label="Thalassemia (thal)",
        choices=[("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)],
        value=2
    )
]

# Define the output component
outputs = gr.Markdown(label="Prediction Result")

# Define example data to show users how to fill the form
examples = [
    [63, "Male", 3, 145, 233, 1, 0, 150, "No", 2.3, 0, 0, 1],
    [37, "Male", 2, 130, 250, 0, 1, 187, "No", 3.5, 0, 0, 2],
    [56, "Female", 1, 140, 294, 0, 0, 153, "No", 1.3, 1, 0, 2],
    [41, "Female", 1, 130, 204, 0, 0, 172, "No", 1.4, 2, 0, 2]
]

# --- Step 5: Launch the Gradio App ---
interface = gr.Interface(
    fn=predict_heart_attack,
    inputs=inputs,
    outputs=outputs,
    title="💖 Ayush's Heart Disease Likelihood Predictor",
    description="### A Machine Learning tool to predict the likelihood of heart disease. \n\nFill in the patient's details below to get a prediction. \n\n**Disclaimer:** This tool is for educational purposes and is not a substitute for professional medical advice. \n\n*Created with ❤️ by Ayush.*",
    examples=examples,
    theme=gr.themes.Soft(),
    allow_flagging="never",  # Disables the flagging feature
    css="footer {visibility: hidden}"  # Hides the "Powered by Gradio" footer for a cleaner look
)

# Add print statements to provide feedback in the terminal before launching
print("Model training complete. Preparing to launch the Gradio interface...")
print("Please copy the LOCAL URL that appears below and paste it into your web browser.")

# Launch the web application. 'share=True' creates a temporary public link.
# 'debug=True' provides more detailed error logs in the terminal if something goes wrong.
interface.launch(share=True, debug=True)

