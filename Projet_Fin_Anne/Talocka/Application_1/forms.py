from django import forms
from .models import Projet_User

class ProjetForm(forms.ModelForm):
    class Meta:
        model = Projet_User
        fields = ['name','description']


from django import forms

class DatasetUploadForm(forms.Form):
    dataset_name = forms.CharField(label="Nom du Dataset", max_length=255)
    description = forms.CharField(label="Description", widget=forms.Textarea, required=False)
    file = forms.FileField(label="Choisir un fichier")
