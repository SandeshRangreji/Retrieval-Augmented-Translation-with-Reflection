import os
import pandas as pd
import sacrebleu


# Function to compute BLEU score for each row using sacrebleu
def compute_bleu_score(reference, translation):
    bleu_score = sacrebleu.sentence_bleu(translation, [reference]).score
    return bleu_score


# Function to analyze translations and write output if BLEU score is below 20
def analyze_translations(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            df = pd.read_csv(input_path)

            # Prepare results for low BLEU score translations
            results = []
            for index, row in df.iterrows():
                source_language = row['source_language']
                target_language = row['target_language']
                source = row['source']
                reference = row['reference']
                translation = row['translation']
                bleu_score = compute_bleu_score(reference, translation)

                # Append to results only if BLEU score is below 20
                if bleu_score < 20:
                    results.append({
                        'source_language': source_language,
                        'target_language': target_language,
                        'source': source,
                        'reference': reference,
                        'translation': translation,
                        'bleu_score': bleu_score
                    })

                    # Print the low BLEU score details
                    print(
                        f"Low BLEU Score:\nSource Language: {source_language}\nTarget Language: {target_language}\nReference: {reference}\nTranslation: {translation}\n")

            # Convert results to DataFrame and save to CSV if not empty
            if results:
                result_df = pd.DataFrame(results)
                output_path = os.path.join(output_folder, f"bleu_analysis_{filename}")
                result_df.to_csv(output_path, index=False)


def main():
    input_folder = 'translated'
    output_folder = 'translation_analysis'
    analyze_translations(input_folder, output_folder)


if __name__ == "__main__":
    main()