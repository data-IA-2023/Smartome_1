from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from .forms import ConnexionForm,InscriptionForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
# Create your views here.


def connexion(request):
    form = ConnexionForm()
    if request.method == 'POST':
        form = ConnexionForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('accueil')
            else:
                form.add_error(None, "Nom d'utilisateur ou mot de passe incorrect.")
                return redirect('accueil')

    return render(request, "connexion.html", {"form": form})

@login_required
def deconnexion(request):
    logout(request)
    return redirect(connexion)

def inscription(request):
    if request.method == 'POST':
        form = InscriptionForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)  
            return redirect('accueil')  
    else:
        form = InscriptionForm()

    return render(request, 'inscription.html', {'form': form})
    

