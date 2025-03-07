from django import forms
from django.contrib.auth.models import User

class ConnexionForm(forms.Form):
    username = forms.CharField(label='Nom d\'utilisateur', max_length=50)
    password = forms.CharField(label='Mot de passe', widget=forms.PasswordInput)



class InscriptionForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, label='Mot de passe')
    password_confirm = forms.CharField(widget=forms.PasswordInput, label='Confirmez le mot de passe')

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")

        if password and password_confirm and password != password_confirm:
            raise forms.ValidationError("Les mots de passe ne correspondent pas.")
        return cleaned_data