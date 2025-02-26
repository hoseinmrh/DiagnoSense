from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import ast
import getpass
from dotenv import load_dotenv
import os


class MatchSymptoms:
    """
    A class to match and find symptoms from user input
    """
    def __init__(self, user_description=None):
        """
        :param user_description: text of user about their feelings to get the symptoms from
        """
        self.known_symptoms = []
        self.prompt_string = """Use the LLM to find semantic matches between user descriptions and known symptoms.\

                Given the following user description of symptoms:
                user_description : {user_description} \

                And the following list of known medical symptoms:
                known_symptoms: {known_symptoms} \
                Identify which symptoms from the known list are present in the user's description.
                Consider synonyms, medical terminology variations, and laymen's terms. \

                Return only a Python list of the matched symptoms, with no additional text or explanation.
                {format_instructions}
                """
        self.llm = None
        self.user_description = user_description
        self.prompt_template = None
        self.format_instructions = None
        self.output_parser = None
        self.user_prompt = None

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

    def set_known_symptoms(self, file_path="../public/features_names.txt"):

        with open(file_path, 'r') as f:
            content = f.read()  # Read the entire file
            self.known_symptoms = [item.strip() for item in content.split(",")]

    def setup_prompt(self):
        result_schema = ResponseSchema(name="result",
                                       description="Match the symptoms and \
                                     output them as a comma separated Python list.")

        response_schemas = [result_schema]

        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.prompt_template = ChatPromptTemplate.from_template(self.prompt_string)
        self.user_prompt = self.prompt_template.format_messages(
            known_symptoms=self.known_symptoms,
            user_description=self.user_description,
            format_instructions=self.format_instructions
        )

    def run(self):
        self.setup_llm()
        self.set_known_symptoms()
        self.setup_prompt()
        response = self.llm.invoke(self.user_prompt)
        response_list_string = self.output_parser.parse(response.content)['result']
        try:
            print(ast.literal_eval(response_list_string))
            return ast.literal_eval(response_list_string)
        except (SyntaxError, ValueError):
            print(f"Error: Could not parse the output as a list. Output was: {list_string}")


model = MatchSymptoms(user_description="My chest heard and I can not breath very well")
model.run()