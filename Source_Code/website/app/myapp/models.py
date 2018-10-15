from django.db import models

# Create your models here.


class User(models.Model):
    email = models.CharField(max_length=60, null=False,
                             unique="True", default="")
    password = models.CharField(max_length=256)

    class Meta:
        db_table = "User"
