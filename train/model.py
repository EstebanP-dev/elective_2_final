from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import os

from preprocess import preprocess_data

def load_data(tweet_file_path, emoji_file_path):
    base_dir = os.path.dirname(__file__)
    
    full_tweet_file_path = tweet_file_path if os.path.isabs(tweet_file_path) else os.path.join(base_dir, tweet_file_path)
    full_emoji_file_path = emoji_file_path if os.path.isabs(emoji_file_path) else os.path.join(base_dir, emoji_file_path)

    if not os.path.exists(full_tweet_file_path):
        raise FileNotFoundError(f"Tweet data file not found: {full_tweet_file_path}")
    
    tweet_data = pd.read_csv(full_tweet_file_path)
    
    emoji_data = pd.DataFrame() 
    if os.path.exists(full_emoji_file_path):
        emoji_data = pd.read_csv(full_emoji_file_path)
    else:
        print(f"Warning: Emoji data file not found at {full_emoji_file_path}. Proceeding without it.")
        emoji_data = pd.DataFrame(columns=['tweet_id', 'sentiment', 'content'])
        
    return tweet_data, emoji_data

def create_model_pipeline():
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2)),
        ('scaler', StandardScaler(with_mean=False)), 
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    return model_pipeline

def save_model_components(fitted_pipeline, encoder, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump({'model_pipeline': fitted_pipeline, 'encoder': encoder}, file)
    print(f"Model components saved to {filename}")

def main():
    tweet_file_rel_path = 'tweet_emotions.csv' 
    emoji_file_rel_path = 'emoji_sentiment_data.csv'

    base_dir = os.path.dirname(__file__)
    dummy_emoji_path = os.path.join(base_dir, emoji_file_rel_path)
    if not os.path.exists(dummy_emoji_path):
        print(f"Creating dummy '{emoji_file_rel_path}' as it was not found. Replace with your actual emoji data.")
        dummy_emoji_df = pd.DataFrame({
            'tweet_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            'sentiment': ['happiness', 'sadness', 'love', 'anger', 'surprise', 'fear', 'neutral', 'enthusiasm', 'worry', 'fun', 'hate', 'boredom', 'empty'],
            'content': [
                'I am so happy :D <3', 
                'Feeling very sad :(', 
                'Love this product!! <3 <3', 
                'This is outrageous! >:( >_<',
                'Wow, I did not expect that! :O',
                'That was a scary movie! :S',
                'Just a regular day.',
                'This is going to be awesome! SOON!',
                "I'm worried about the test results.",
                "Having a great time lol :P",
                "I hate waiting in line so much.",
                "Nothing to do, so bored.",
                ""
            ]
        })
        dummy_emoji_df.to_csv(dummy_emoji_path, index=False)
    
    raw_tweet_data, raw_emoji_data = load_data(tweet_file_rel_path, emoji_file_rel_path)
    
    try:
        train_df, val_df = preprocess_data(raw_tweet_data, raw_emoji_data)
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        return

    if train_df.empty:
        print("Error: Training data is empty after preprocessing. Check input files and preprocessing logic.")
        return
    
    X_train_text = train_df['content']
    y_train_labels = train_df['sentiment']
    
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_labels)
    
    model_pipeline = create_model_pipeline()
    
    print(f"Starting model training with {len(X_train_text)} samples...")
    model_pipeline.fit(X_train_text, y_train_encoded)
    print("Model training complete.")

    if not val_df.empty:
        X_val_text = val_df['content']
        y_val_labels = val_df['sentiment']
        if len(X_val_text) > 0:
            y_val_encoded_actual = encoder.transform(y_val_labels)
            y_val_pred_encoded = model_pipeline.predict(X_val_text)
            
            known_labels_mask = [label for label in y_val_labels if label in encoder.classes_]
            y_val_pred_labels = encoder.inverse_transform(y_val_pred_encoded)

            print("\nValidation Set Performance:")

            report_labels = sorted(list(set(y_val_labels) | set(y_val_pred_labels)))
            target_names_for_report = [label for label in report_labels if label in encoder.classes_]
            
            if not target_names_for_report:
                 print("Could not generate classification report: No common labels found or encoder issues.")
            else:
                print(classification_report(y_val_labels, y_val_pred_labels, labels=target_names_for_report, zero_division=0))
        else:
            print("Validation set is empty or has no content to evaluate.")
    else:
        print("Validation DataFrame is empty. No evaluation performed.")


    model_save_path = os.path.join(base_dir, 'sentiment_model.pkl')
    save_model_components(model_pipeline, encoder, model_save_path)

if __name__ == "__main__":
    main()