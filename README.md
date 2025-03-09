# DiagnoSense

DiagnoSense is a medical diagnostic assistant that uses AI to predict diseases based on user symptoms and provides human-readable explanations for the predictions.

## Introduction

Machine learning models are increasingly used in healthcare, but their predictions often lack transparency. This project aims to enhance **ML explainability** by developing an interpretable disease prediction model.

### Key Features
- **Disease Prediction**: A **CatBoost** model trained on **132 symptoms** to classify **41 diseases**.
- **Explainability Layer**: Predictions include confidence scores, supporting and contradicting symptoms, and a full probability breakdown using **SHAP**.
- **LLM Integration**: A **Gemini LLM** generates human-readable explanations for the predictions and helps users describe their symptoms in natural language, mapping them to structured data.

### Technologies Used
- 🧠 **LLM (Gemini)** – for explanation generation and symptom mapping
- 📊 **CatBoost** – for disease classification
- 🔍 **SHAP** – for feature importance analysis

This approach ensures that AI remains a **decision-support tool**, keeping humans in control of healthcare decisions.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/DiagnoSense.git
    cd DiagnoSense
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

### Matching Symptoms

The `MatchSymptoms` class is used to match and find symptoms from user input.

```python
from match_symptoms import MatchSymptoms

match_symptoms_model = MatchSymptoms(user_description="I have headache all the time and when I try to wake up, I feel dizzy")
user_symptoms = match_symptoms_model.run()
```

### Disease Prediction

The `DiseasePredictor` class is used to predict diseases based on symptoms and provide explanations.

```python
from disease_predictor import DiseasePredictor
import pandas as pd

# Load training data
train_df = pd.read_csv('../public/training_data.csv')
train_x = train_df.iloc[:, 0:132]  # Symptoms
train_y = train_df.iloc[:, 132]    # Disease labels

# Predict disease
new_predictor = DiseasePredictor()
new_predictor.load_model()
new_predictor.feature_names = train_x.columns.tolist()
model_feature_names = new_predictor.feature_names
user_input_df = new_predictor.create_model_input(user_symptoms, model_feature_names)
explanation = new_predictor.explain_prediction(user_input_df)
new_predictor.print_explanation(explanation)
```

### Generating Explainability

The `GenerateExplainability` class is used to generate human-readable explanations and save them as PDFs.

```python
from generate_explainability import GenerateExplainability

# Generate explanation and save as PDF
generate_explainability_model = GenerateExplainability()
result = generate_explainability_model.run(explanation, user_symptoms)
```

### Full Example

Here's a full example of how to use the classes together:

```python
from match_symptoms import MatchSymptoms
from disease_predictor import DiseasePredictor
from generate_explainability import GenerateExplainability
import pandas as pd

# Match symptoms
match_symptoms_model = MatchSymptoms(user_description="I have headache all the time and when I try to wake up, I feel dizzy")
user_symptoms = match_symptoms_model.run()

# Load training data
train_df = pd.read_csv('../public/training_data.csv')
train_x = train_df.iloc[:, 0:132]  # Symptoms
train_y = train_df.iloc[:, 132]    # Disease labels

# Predict disease
new_predictor = DiseasePredictor()
new_predictor.load_model()
new_predictor.feature_names = train_x.columns.tolist()
model_feature_names = new_predictor.feature_names
user_input_df = new_predictor.create_model_input(user_symptoms, model_feature_names)
explanation = new_predictor.explain_prediction(user_input_df)

# Generate explanation and save as PDF
generate_explainability_model = GenerateExplainability()
result = generate_explainability_model.run(explanation, user_symptoms)
```