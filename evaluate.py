import sacrebleu
from comet import download_model, load_from_checkpoint
import os
import pandas as pd

base_dir = '/Users/sandeshrangreji/PycharmProjects/Retrieval-Augmented-Translation-with-Reflection/translated'
language_pairs = ['en-de', 'en-fr', 'en-pt']

# Load COMET-22 model
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)


def evaluate_translations(df):
    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(df['translation'].tolist(), [df['reference'].tolist()])
    bleu_score = bleu.score

    # Prepare data for COMET
    comet_data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(df['source'], df['translation'], df['reference'])
    ]

    # Compute COMET-22 scores
    comet_scores = comet_model.predict(comet_data, batch_size=8, gpus=0)

    # Extract COMET scores
    if isinstance(comet_scores, dict) and 'scores' in comet_scores:
        comet_scores = comet_scores['scores']
    elif hasattr(comet_scores, 'scores'):
        comet_scores = comet_scores.scores
    else:
        print("Unexpected structure in comet_scores:", comet_scores)
        raise ValueError("Cannot extract 'scores' from comet_scores")

    # Calculate the average COMET score
    comet_score = sum(comet_scores) / len(comet_scores)

    return bleu_score, comet_score


# Evaluate translations on the validation set
for lang_pair in language_pairs:
    file_path = os.path.join(base_dir, f'{lang_pair}_valid_translated_subset.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty:
            bleu_score, comet_score = evaluate_translations(df)
            print(f"Language Pair: {lang_pair}")
            print(f"BLEU Score: {bleu_score}")
            print(f"COMET-22 Score: {comet_score}")
        else:
            print(f"The file {file_path} is empty.")
    else:
        print(f"File {file_path} does not exist.")
