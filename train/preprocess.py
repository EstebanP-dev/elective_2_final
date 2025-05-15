from sklearn.model_selection import train_test_split
import pandas as pd
import re
import emoji

def clean_text(text):
    
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = text.strip()
    return text

def handle_emojis(text):
    if not isinstance(text, str):
        return ""
    
    return emoji.demojize(text, delimiters=(" ", " "))

def preprocess_data(tweets_input_df, emojis_input_df):
    tweets_df = tweets_input_df.copy()
    emojis_df = emojis_input_df.copy()

    if 'content' in tweets_df.columns:
        tweets_df['content'] = tweets_df['content'].astype(str).apply(clean_text)
        tweets_df['content'] = tweets_df['content'].apply(handle_emojis)
    else:
        raise ValueError("Tweet DataFrame must contain a 'content' column.")

    if not emojis_df.empty:
        if 'content' in emojis_df.columns:
            emojis_df['content'] = emojis_df['content'].astype(str).apply(clean_text)
            emojis_df['content'] = emojis_df['content'].apply(handle_emojis)
        else:
            print("Warning: Emoji DataFrame provided but no 'content' column found for preprocessing.")

    if 'sentiment' not in tweets_df.columns:
        raise ValueError("Tweet DataFrame must have a 'sentiment' column.")
    
    if not emojis_df.empty and 'sentiment' not in emojis_df.columns:
        raise ValueError("Emoji DataFrame must have a 'sentiment' column if not empty.")

    combined_df = pd.concat([tweets_df, emojis_df], ignore_index=True)

    combined_df.dropna(subset=['sentiment', 'content'], inplace=True)
    combined_df = combined_df[combined_df['content'].str.strip() != '']

    if combined_df.empty:
        raise ValueError("Combined DataFrame is empty after preprocessing and cleaning. Check input data.")

    if combined_df['sentiment'].nunique() > 1:
        try:
            train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['sentiment'])
        except ValueError as e:
            print(f"Stratify failed: {e}. Splitting without stratify. Consider checking class distribution.")
            train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    else:
        print("Warning: Only one class present in sentiment column. Cannot stratify. Consider diversifying data.")
        train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

    return train_df, val_df