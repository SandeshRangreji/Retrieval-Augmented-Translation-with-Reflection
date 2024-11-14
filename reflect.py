from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the GPT-4o model for evaluation
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))


# Define the Pydantic model for structured feedback output
class EvaluationOutput(BaseModel):
    feedback: str = Field(description="Detailed feedback on the quality of the translation.")


# Initialize the structured output parser for evaluation
parser = PydanticOutputParser(pydantic_object=EvaluationOutput)

# Create a prompt template for evaluating translations
evaluation_prompt = PromptTemplate(
    template=(
        "Evaluate the translation provided below and give feedback on how to improve it if necessary. "
        "{format_instructions}\n"
        "Original Sentence: {original_sentence}\n"
        "Provided Translation: {provided_translation}\n"
        "Evaluation:"
    ),
    input_variables=["original_sentence", "provided_translation"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# Function to evaluate a translation and provide feedback
def evaluate_translation(original_sentence, provided_translation):
    # Construct the formatted input
    formatted_input = evaluation_prompt.format(
        original_sentence=original_sentence,
        provided_translation=provided_translation
    )

    # Invoke the model and parse the output
    response = llm.invoke([{"role": "user", "content": formatted_input}])
    parsed_response = parser.parse(response.content)
    return parsed_response.feedback


# Main function to test the evaluation chain with hardcoded examples
def main():
    # Hardcoded test examples
    test_examples = [
        {
            "original_sentence": "The cat is sitting on the mat.",
            "provided_translation": "Die Katze sitzt auf der Matte."
        },
        {
            "original_sentence": "I love programming and learning new technologies.",
            "provided_translation": "Ich liebe es zu programmieren und neue Technologien zu lernen."
        },
        {
            "original_sentence": "The weather today is perfect for a walk in the park.",
            "provided_translation": "Das Wetter heute ist perfekt für einen Spaziergang im Park."
        }
    ]

    # Evaluate each example
    for i, example in enumerate(test_examples):
        feedback = evaluate_translation(
            example["original_sentence"],
            example["provided_translation"]
        )

        # Print results for manual inspection
        print(f"\nExample {i + 1}:")
        print(f"Original Sentence: {example['original_sentence']}")
        print(f"Provided Translation: {example['provided_translation']}")
        print(f"Feedback: {feedback}")


# Run the main function to test the evaluation chain
if __name__ == "__main__":
    main()