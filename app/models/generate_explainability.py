from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import ast
import getpass
from dotenv import load_dotenv
import os
from match_symptoms import MatchSymptoms
from disease_predictor import DiseasePredictor
import pandas as pd


class GenerateExplainability:
    """
    A class to generate human readable explainability of the prediction and the decision behind that
    """
    def __init__(self):
        """

        :param explanation: the explanation of the prediction from catboost
        :param detected_disease: the predicted disease by model
        :param user_symptoms: the symptoms of the user
        """
        self.llm = None
        self.prompt_string = """
You are a medical assistant explaining diagnostic predictions to patients. 
I'll provide you with the output from a medical diagnostic AI system and your job is to explain it clearly and conversationally to the patient.

Here's the prediction information:
- Predicted disease: {predicted_disease}
- Confidence level: {confidence}
- Supporting symptoms and their importance: {supporting_symptoms}
- Contradicting symptoms: {contradicting_symptoms}
- Alternative possible diagnoses and their probabilities: {alternative_diagnoses}

Patient's reported symptoms: {patient_symptoms}

Please create a clear, compassionate explanation that:
1. States the predicted condition in simple terms
2. Explains which symptoms support this diagnosis and why. Only use the symptoms predicted for this diagnosis.
3. Mentions any symptoms that don't typically match this condition (if any)
4. Mentions other possible conditions that might be considered
5. Reminds the patient that this is an AI-generated assessment and they should consult a healthcare professional

Use conversational language that a non-medical person would understand. Avoid technical jargon and use analogies where helpful.
"""

    def setup_llm(self):
        load_dotenv()
        apiKey = os.getenv('GOOGLE_API_KEY')
        os.environ["GOOGLE_API_KEY"] = apiKey

        self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    )
        print("model setup complete ge")

    def generate_human_explanation(self, explanation_dict, patient_symptoms):
        """
        Generate a human-friendly explanation using LLM based on the model's explanation.

        Args:
            explanation_dict: Dictionary returned from explain_prediction function
            patient_symptoms: List of symptoms the patient reported having

        Returns:
            A string with human-friendly explanation from the LLM
        """
        if self.llm is None:
            self.setup_llm()

        # Format the alternative diagnoses
        all_probs = explanation_dict["all_probabilities"]
        pred_disease = explanation_dict["predicted_disease"]

        # Sort and get top 3 alternative diagnoses
        alternative_diagnoses = [(disease, prob) for disease, prob in all_probs.items()
                                 if disease != pred_disease]
        alternative_diagnoses.sort(key=lambda x: x[1], reverse=True)
        alternative_diagnoses = alternative_diagnoses[:3]

        # Format alternative diagnoses for prompt
        alt_diag_formatted = []
        for disease, prob in alternative_diagnoses:
            alt_diag_formatted.append(f"({disease}, {prob * 100:.2f}%)")

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template(self.prompt_string)

        # Format messages for LLM
        messages = prompt_template.format_messages(
            predicted_disease=explanation_dict["predicted_disease"],
            confidence=explanation_dict["confidence"],
            supporting_symptoms=explanation_dict["supporting_symptoms"],
            contradicting_symptoms=explanation_dict["contradicting_symptoms"],
            alternative_diagnoses=alt_diag_formatted,
            patient_symptoms=", ".join(patient_symptoms)
        )

        # Get response from LLM
        response = self.llm.invoke(messages)
        return response.content

    def run(self, explanation_dict, patient_symptoms):
        self.setup_llm()
        return self.generate_human_explanation(explanation_dict, patient_symptoms)


match_symptoms_model = MatchSymptoms(user_description="I have headache all the time and when I try to wake up, I feel "
                                                      "dizzy")
user_symptoms = match_symptoms_model.run()

train_df = pd.read_csv('../public/training_data.csv')
train_x = train_df.iloc[:, 0:132]  # Symptoms
train_y = train_df.iloc[:, 132]    # Disease labels

new_predictor = DiseasePredictor()
new_predictor.load_model()

new_predictor.feature_names = train_x.columns.tolist()
model_feature_names = new_predictor.feature_names

user_input_df = new_predictor.create_model_input(user_symptoms, model_feature_names)

explanation = new_predictor.explain_prediction(user_input_df)

generate_explainability_model = GenerateExplainability()
result = generate_explainability_model.run(explanation, user_symptoms)

print(result)

