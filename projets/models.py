from django.db import models

class Projet(models.Model):
    text=models.TextField()
    fake_proba=models.FloatField()
    real_proba=models.FloatField()
