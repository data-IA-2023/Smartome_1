from django.shortcuts import render
from django.shortcuts import render, redirect
from datetime import datetime
# from .forms import ConnexionForm,InscriptionForm
from .forms import ProjetForm,DatasetUploadForm
from .models import Projet_User,DatasetMetadata
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import get_object_or_404
from .utils import get_mongo_gridfs
from bson import ObjectId

@login_required
def accueil(request):
    return render(request, "accueil.html")

@login_required
def projets(request):
    form = ProjetForm()
    projets_list = Projet_User.objects.filter(utilisateur=request.user)
    return render(request,'projets.html',{"form":form,"projets_list":projets_list})

@login_required
def create_projet(request):
    if request.method == 'POST':
        form = ProjetForm(request.POST)
        if form.is_valid():
            projet_name = form.cleaned_data['name']
            if Projet_User.objects.filter(name=projet_name, utilisateur=request.user).exists():  
                messages.error(request, f"Un projet avec le nom: {projet_name} existe déjà")  
                return redirect('projets')
            else:   
                try:   
                    projet = form.save(commit=False) 
                    projet.utilisateur = request.user 
                    projet.date_de_creation = datetime.now().date()
                    projet.date_de_modification = datetime.now().date()
                    projet.save()  
                    messages.success(request, f"Projet créé avec succès!")
                    return redirect('projets')
                except Exception as e:
                    messages.error(request, f"Erreur pendant la création du projet, {e}")
                    return redirect('projets')
        else:
            messages.error(request,"Erreur lors du chargement du formulaire")  
            return redirect('projets')  
    return redirect('projets')

@login_required
def modifier_projet(request, projet_id):
    projet = get_object_or_404(Projet_User, id=projet_id, utilisateur=request.user)

    if request.method == "POST":
        form = ProjetForm(request.POST)  
        return render(request, 'modifier_projet.html', {'form': form, 'projet': projet})

    else:
        return redirect('projets') 

@login_required
def modification(request,projet_id) :
    projet = get_object_or_404(Projet_User, id=projet_id, utilisateur=request.user)
    if request.method == 'POST':
        form = ProjetForm(request.POST, instance=projet) 
        if form.is_valid():
            form.save()
            messages.success(request, "Projet modifié avec succès !")
            return redirect('projets')  

        else:
            messages.error(request, "Erreur lors de la modification du projet. Veuillez corriger les erreurs ci-dessous.")
            return render(request, 'modifier_projet.html', {'form': form, 'projet': projet})

@login_required
def delete_projet(request):
    if request.method == 'POST':
        Supprimer = request.POST.get('Supprimer',None)
        if Supprimer:
            projet = get_object_or_404(Projet_User, id=Supprimer, utilisateur=request.user)
            projet.delete()
            messages.success(request, "Projet supprimé avec succès !")
            return redirect('projets')  
        else:
            messages.success(request, "Projet non trouvé")
            return redirect('projets')  

    return redirect('projets')
@login_required
def upload_dataset(request, projet_id):
    form = DatasetUploadForm()
    projet = Projet_User.objects.get(id=projet_id, utilisateur=request.user)
    datasets = DatasetMetadata.objects.filter(projet=projet)
    if request.method == 'POST': 
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset_name = form.cleaned_data['dataset_name']
            if DatasetMetadata.objects.filter(projet=projet, dataset_name=dataset_name).exists():
                messages.error(request, f"Un dataset nommé '{dataset_name}' existe déjà !")
                return render(request, "upload_dataset.html", {"form": form, "projet": projet, "datasets": datasets})
            description = form.cleaned_data['description']
            uploaded_file = request.FILES['file']
            db,grid_fs = get_mongo_gridfs()
            file_id = grid_fs.put(uploaded_file, filename=dataset_name)
            DatasetMetadata.objects.create(
                projet=projet,
                dataset_name=dataset_name,
                description=description,
                file_id=str(file_id))
            messages.success(request, "Dataset téléchargé avec succès !")
            return redirect("upload_dataset", projet_id=projet_id)

        else:
            if form.is_valid():
                messages.error(request, "Erreur lors de la soumission du formulaire")
            else:
                return render(request, "upload_dataset.html", context={"form": form, "projet": projet, "datasets": datasets})

    return render(request, "upload_dataset.html", context={"form": form, "projet": projet, "datasets": datasets})

@login_required
def delete_dataset(request, dataset_id):
    dataset = get_object_or_404(DatasetMetadata, id=dataset_id, projet__utilisateur=request.user)

    if request.method == "POST":
        db,grid_fs = get_mongo_gridfs()

        try:
            file_id = ObjectId(dataset.file_id)  
        except Exception as e:
            messages.error(request, f"Erreur lors de la conversion de l'ID du fichier MongoDB : {e}")
            return redirect("upload_dataset", projet_id=dataset.projet.id)

        try:
            grid_fs.delete(file_id)
        except Exception as e:
            messages.error(request, f"Erreur lors de la suppression du fichier MongoDB : {e}")
            return redirect("upload_dataset", projet_id=dataset.projet.id)
        try:
            db.fs.chunks.delete_many({"files_id": file_id})
        except Exception as e:
            messages.error(request, f"Erreur lors de la suppression des chunks MongoDB : {e}")
            return redirect("upload_dataset", projet_id=dataset.projet.id)

        try:
            dataset.delete()
        except Exception as e:
            messages.error(request, f"Erreur lors de la suppression du dataset en base de données : {e}")
            return redirect("upload_dataset", projet_id=dataset.projet.id)

        messages.success(request, "Dataset supprimé avec succès !")
        return redirect("upload_dataset", projet_id=dataset.projet.id)

    return redirect("upload_dataset", projet_id=dataset.projet.id)

