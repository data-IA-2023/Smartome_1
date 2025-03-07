from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from Application_1.models import Projet_User

class ProjetTests(TestCase):

    def setUp(self):
        """Créer un utilisateur et un projet de test"""
        self.user = User.objects.create_user(username='testuser', password='testpassword')
        self.client.login(username='testuser', password='testpassword')

        # Création d'un projet existant
        self.projet = Projet_User.objects.create(
            name="Projet Existant",
            description="Description du projet existant.",
            utilisateur=self.user
        )

    def test_projets_view(self):
        """Tester l'affichage de la page des projets"""
        response = self.client.get(reverse('projets'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<form")  # Vérifie que le formulaire est présent
        self.assertContains(response, "Projet Existant")  # Vérifie que le projet existant est affiché

    def test_create_projet_view(self):
        """Tester la création d'un projet"""
        response = self.client.post(reverse('create_projet'), {
            'name': 'Projet Test',
            'description': 'Ceci est un projet de test.'
        })

        # Vérifier la redirection après création
        self.assertEqual(response.status_code, 302)  
        
        # Vérifier que le projet a bien été créé
        self.assertTrue(Projet_User.objects.filter(name='Projet Test', utilisateur=self.user).exists())

    def test_create_duplicate_projet(self):
        """Tester la création d'un projet en double (même nom)"""
        response = self.client.post(reverse('create_projet'), {
            'name': 'Projet Existant',  # Nom déjà utilisé
            'description': 'Tentative de création en double'
        })

        # Vérifier la redirection après erreur
        self.assertEqual(response.status_code, 302)  
        
        # Vérifier que le projet en double n'a pas été créé
        self.assertEqual(Projet_User.objects.filter(name="Projet Existant", utilisateur=self.user).count(), 1)

