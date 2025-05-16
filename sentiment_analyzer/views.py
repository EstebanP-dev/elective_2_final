from django.shortcuts import render
from django.views.generic import FormView
from django.urls import reverse_lazy
from .forms import SentimentForm
from .predictor import SentimentPredictor


class SentimentAnalysisView(FormView):
    template_name = 'sentiment_analyzer/analyze.html'
    form_class = SentimentForm
    success_url = reverse_lazy('sentiment_analysis')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictor = SentimentPredictor()

    def form_valid(self, form):
        # Get the comment from the form
        comment = form.cleaned_data['comment']

        # Get sentiment prediction
        result = self.predictor.predict_sentiment(comment)

        # Add result to context
        return self.render_to_response(
            self.get_context_data(
                form=form,
                result=result,
                comment=comment
            )
        )


def home(request):
    return render(request, 'sentiment_analyzer/home.html')