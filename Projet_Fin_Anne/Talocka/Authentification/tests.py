from django.test import TestCase


from django.contrib.auth.models import User
from django.urls import reverse

class AuthTests(TestCase):

    def setUp(self):
        """Créer un utilisateur pour les tests"""
        self.user = User.objects.create_user(username='testuser', password='testpassword')

    def test_inscription_view(self):
        """Tester la vue d'inscription"""
        response = self.client.post(reverse('inscription'), {
            'username': 'nouvelutilisateur',
            'email': 'nouvelutilisateur@example.com',
            'password': 'testpassword',
            'password_confirm': 'testpassword'
        })
        self.assertEqual(response.status_code, 302)  
        self.assertTrue(User.objects.filter(username='nouvelutilisateur').exists())

    def test_connexion_view(self):
        """Tester la connexion avec un utilisateur existant"""
        response = self.client.post(reverse('connexion'), {
            'username': 'testuser',
            'password': 'testpassword'
        })
        self.assertEqual(response.status_code, 302)  

    def test_deconnexion_view(self):
        """Tester la déconnexion"""
        self.client.login(username='testuser', password='testpassword')
        response = self.client.get(reverse('deconnexion'))
        self.assertEqual(response.status_code, 302)  
