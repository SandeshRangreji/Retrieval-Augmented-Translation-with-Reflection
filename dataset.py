import os
import pandas as pd

base_dir = '/Users/sandeshrangreji/PycharmProjects/Retrieval-Augmented-Translation-with-Reflection/chat-task-2024-data'
language_pairs = ['en-de', 'en-fr', 'en-pt']
splits = ['train', 'valid']


def load_and_preprocess_dataset(split, lang_pair):
    file_path = os.path.join(base_dir, split, f'{lang_pair}.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        source_lang, target_lang = lang_pair.split('-')

        # Replace all variations of "pt-" with "pt" in source and target languages
        df['source_language'] = df['source_language'].apply(lambda x: 'pt' if x.startswith('pt-') else x)
        df['target_language'] = df['target_language'].apply(lambda x: 'pt' if x.startswith('pt-') else x)

        # Identify rows where the language direction is flipped
        flipped_rows = (df['source_language'] == target_lang) & (df['target_language'] == source_lang)

        # Flip the language columns and content for those rows
        if flipped_rows.any():
            df.loc[flipped_rows, ['source_language', 'target_language']] = df.loc[flipped_rows, ['target_language', 'source_language']].values
            df.loc[flipped_rows, ['source', 'reference']] = df.loc[flipped_rows, ['reference', 'source']].values

        # Keep only rows that are in the form `en-<target>`
        correct_direction_rows = (df['source_language'] == source_lang) & (df['target_language'] == target_lang)
        df = df[correct_direction_rows]

        return df
    else:
        print(f"File {file_path} does not exist.")
        return None


# Create the datasets dictionary
datasets = {}
for lang_pair in language_pairs:
    datasets[lang_pair] = {}
    for split in splits:
        df = load_and_preprocess_dataset(split, lang_pair)
        if df is not None:
            datasets[lang_pair][split] = df
            print(f"{split} data for {lang_pair} loaded and preprocessed with {len(df)} rows.")
        else:
            datasets[lang_pair][split] = pd.DataFrame()
            print(f"{split} data for {lang_pair} is empty or could not be loaded.")
