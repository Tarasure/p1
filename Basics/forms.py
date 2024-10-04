from django import forms
from .models import PhishingDataset
class PhishingForm(forms.ModelForm):
    class Meta:
        model = PhishingDataset
        fields = ('url',)

    def _init_(self, *args, **kwargs):
        super(PhishingForm, self)._init_(*args, **kwargs)
        self.fields['url'].widget.attrs['placeholder'] = 'Enter a URL to check for phishing'

    def clean_url(self):
        url = self.cleaned_data['url']
        if not url.startswith('http'):
            raise forms.ValidationError('Please enter a valid URL starting with http or https')
        return url 