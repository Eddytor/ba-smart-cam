from django import forms
from .models import ImageFile


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = ImageFile
        fields = ('title', 'image')