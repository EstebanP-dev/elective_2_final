import pickle
import os
from pathlib import Path


class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.load_model()

    def load_model(self):
        try:
            # Get the path to the model file
            base_dir = Path(__file__).resolve().parent.parent
            model_path = os.path.join(base_dir, 'train', 'sentiment_model.pkl')

            with open(model_path, 'rb') as model_file:
                model_data = pickle.load(model_file)
                self.model_pipeline = model_data['model_pipeline']
                self.encoder = model_data['encoder']

            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict_sentiment(self, text):
        if not self.model_pipeline or not self.encoder:
            return "Error: Model not loaded"

        try:
            # Predict sentiment
            prediction_encoded = self.model_pipeline.predict([text])[0]
            prediction = self.encoder.inverse_transform([prediction_encoded])[0]

            # Map sentiment to feedback
            feedback = self.get_feedback(prediction)

            return {
                'sentiment': prediction,
                'feedback': feedback
            }
        except Exception as e:
            return {
                'sentiment': 'Error',
                'feedback': f'Analysis failed: {str(e)}'
            }

    def get_feedback(self, sentiment):
        feedback_map = {
            'happiness': "That's great to hear! 😊",
            'sadness': "I'm sorry you feel that way. 😔",
            'love': "Wonderful! Spread the love! ❤️",
            'anger': "Take a deep breath. Maybe things will get better. 😤",
            'surprise': "Wow! That's unexpected! 😮",
            'fear': "Don't worry, everything will be alright. 😨",
            'neutral': "Thanks for sharing your thoughts.",
            'enthusiasm': "Your excitement is contagious! 🎉",
            'worry': "Try not to overthink it. It'll be okay. 😟",
            'fun': "Sounds like you're having a great time! 😄",
            'hate': "I hope things improve for you soon. 😠",
            'boredom': "Maybe try something new? 😴",
        }

        return feedback_map.get(sentiment.lower(), "Thanks for sharing your thoughts!")