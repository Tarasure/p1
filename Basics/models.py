from django.db import models
class PhishingDataset(models.Model):
    url = models.CharField(max_length=255)
    label = models.IntegerField()
    def _str_(self):
        return self.url
class CyberbullyingDataset(models.Model):
    text = models.TextField()
    label = models.IntegerField()
    def _str_(self):
        return self.text
class PhishingResult(models.Model):
    url = models.CharField(max_length=255)
    result = models.CharField(max_length=255)
    def _str_(self):
        return self.url
class CyberbullyingResult(models.Model):
    text = models.TextField()
    result = models.CharField(max_length=255)
    def _str_(self):
        return self.text
