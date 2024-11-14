from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dataset import datasets
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

language_pairs = ['en-de', 'en-fr', 'en-pt']
n = 100  # Number of sentences to translate per language pair


# Initialize the GPT-4o model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))


# Define a dictionary mapping ISO 639-1 codes to full language names
iso_639_1_to_language = {
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'pt': 'Portuguese',
    'ko': 'Korean',
    'nl': 'Dutch',
    # Add other language codes and their corresponding names as needed
}


# Define the Pydantic model for structured output
class TranslationOutput(BaseModel):
    translated_sentence: str = Field(description="The translated sentence.")


# Initialize the structured output parser
parser = PydanticOutputParser(pydantic_object=TranslationOutput)

# Create a prompt template with format instructions for the model
prompt = PromptTemplate(
    template="Translate the sentence from {source_language} to {target_language}.\n{format_instructions}\nInput Sentence: {input_sentence}\n Translation:",
    input_variables=["input_sentence", "source_language", "target_language"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# Function to run the translation process
def translate_single_sentence(sentence, source_lang, target_lang):
    source_lang = iso_639_1_to_language.get(source_lang, 'Unknown')
    target_lang = iso_639_1_to_language.get(target_lang, 'Unknown')
    # Construct the formatted input
    formatted_input = prompt.format(
        input_sentence=sentence,
        source_language=source_lang,
        target_language=target_lang
    )

    # Invoke the model and parse the output
    response = llm.invoke([{"role": "user", "content": formatted_input}])
    parsed_response = parser.parse(response.content)
    return parsed_response.translated_sentence


# Example usage for translating a validation set
for lang_pair in language_pairs:
    valid_df = datasets[lang_pair]['valid']
    if not valid_df.empty:
        source_code, target_code = lang_pair.split('-')
        translations = []
        print(f"Translating {lang_pair} validation set...")
        valid_subset_df = valid_df.sample(frac=1, random_state=42).copy()[:n]  # Shuffle and take the first 80 rows
        for i in tqdm(range(len(valid_subset_df)), desc=f"Processing {lang_pair}"):
            sentence = valid_subset_df['source'].iloc[i]
            translated_sentence = translate_single_sentence(sentence, source_code, target_code)
            translations.append(translated_sentence)
        valid_subset_df['translation'] = translations
        output_path = os.path.join("", 'translated', f'{lang_pair}_valid_translated_subset.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        valid_subset_df.to_csv(output_path, index=False)
        print(f"Saved translated {lang_pair} validation subset to {output_path}")
