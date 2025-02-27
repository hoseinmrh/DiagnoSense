from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import ast
import getpass
from dotenv import load_dotenv
import os


class GenerateExplainability:
    """
    A class to generate human readable explainability of the prediction and the decision behind that
    """
    def __init__(self, explanation:None, detected_disease:None, user_symptoms:None):
        """

        :param explanation: the explanation of the prediction from catboost
        :param detected_disease: the predicted disease by model
        :param user_symptoms: the symptoms of the user
        """
        self.explanation = explanation
        self.llm = None
        self.prompt_string = []
        self.user_symptoms = user_symptoms
        self.detected_disease = detected_disease

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
        print("model setup complete")
