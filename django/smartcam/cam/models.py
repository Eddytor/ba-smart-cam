from django.db import models

class ImageFile(models.Model):
    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images')

    def __str__(self):
        return self.title


class ImageView(models.Model):
    image = models.ImageField()

    def __str__(self):
        return self.image.name