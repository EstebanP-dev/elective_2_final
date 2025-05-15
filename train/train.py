from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

tweets_df = pd.read_csv('tweet_emotions.csv')
emoji_df = pd.read_csv('emoji_sentiment_data.csv')

combined_df = pd.concat([tweets_df, emoji_df], ignore_index=True)

X = combined_df['content']
y = combined_df['sentiment']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])


pipeline.fit(X_train.values.reshape(-1, 1), y_train)

with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump({
        'model': pipeline,
        'scaler': StandardScaler(),
        'encoder': label_encoder
    }, model_file)