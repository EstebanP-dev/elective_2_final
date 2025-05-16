from django.urls import path
from .views import SentimentAnalysisView, home

urlpatterns = [
    path('', home, name='home'),
    path('analyze/', SentimentAnalysisView.as_view(), name='sentiment_analysis'),
]