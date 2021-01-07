from django.db import models
from django.core.validators import int_list_validator
# Create your models here.
class user_data(models.Model):
    text = models.CharField(max_length=5000)
    total_result = models.IntegerField(default = 0,null=False)
    Sent_result = models.CharField(validators=[int_list_validator],max_length=1000,null=False)