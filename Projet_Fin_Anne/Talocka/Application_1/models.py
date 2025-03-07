from django.db import models

class Projet_User(models.Model):
    name = models.CharField(max_length=50)
    description = models.TextField(max_length=200)
    date_de_creation = models.DateField(auto_now_add=True)
    date_de_modification = models.DateField(auto_now=True)
    utilisateur = models.ForeignKey('auth.User', on_delete=models.CASCADE)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['name', 'utilisateur'], name='unique_name_per_user')
        ]


class DatasetMetadata(models.Model):
    projet = models.ForeignKey(Projet_User, on_delete=models.CASCADE)
    dataset_name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    file_id = models.CharField(max_length=255)  
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.dataset_name} ({self.file_id})"
