# This script loads ai_text.txt and human_text.txt, prepares the dataset, and saves it as a CSV for ML training.
import pandas as pd

# Paths to your data files
ai_path = '../css/js/assets/backend/data/ai_text.txt'
human_path = '../css/js/assets/backend/data/human_text.txt'

# Read AI-generated text
with open(ai_path, 'r', encoding='utf-8') as f:
    ai_texts = [line.strip() for line in f if line.strip()]

# Read human-written text
with open(human_path, 'r', encoding='utf-8') as f:
    human_texts = [line.strip() for line in f if line.strip()]

# Create DataFrame
texts = ai_texts + human_texts
labels = ['ai'] * len(ai_texts) + ['human'] * len(human_texts)
df = pd.DataFrame({'text': texts, 'label': labels})

# Shuffle the dataset
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV for training
shuffled_df.to_csv('dataset.csv', index=False)
print('Dataset prepared and saved as dataset.csv')
