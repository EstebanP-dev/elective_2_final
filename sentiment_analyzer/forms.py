from django import forms

class SentimentForm(forms.Form):
    comment = forms.CharField(
        label='Enter your comment',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Type your comment here...',
            'rows': 4,
        }),
        required=True
    )