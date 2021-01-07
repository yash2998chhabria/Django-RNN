from django.db import models

# Create your models here.
class user_data(models.Model):
    text = models.models.CharField(max_length=5000)
    total_result = models.IntegerField(default = 0,null=False)
    Sent_result = models.CharField(max_length=1000,null=False)