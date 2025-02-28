# DiagnoSense


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

---