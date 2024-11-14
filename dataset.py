import os
import pandas as pd

base_dir = '/Users/sandeshrangreji/PycharmProjects/Retrieval-Augmented-Translation-with-Reflection/chat-task-2024-data'
language_pairs = ['en-de', 'en-fr', 'en-pt', 'en-ko', 'en-nl']
splits = ['train', 'valid', 'test']


def load_dataset(split, lang_pair):
    file_path = os.path.join(base_dir, split, f'{lang_pair}.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File {file_path} does not exist.")
        return None


datasets = {}
for lang_pair in language_pairs:
    datasets[lang_pair] = {}
    for split in splits:
        df = load_dataset(split, lang_pair)
        if df is not None:
            datasets[lang_pair][split] = df
        else:
            datasets[lang_pair][split] = pd.DataFrame()
