from django.db import models

class VideoScan(models.Model):
    score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Scan {self.id} - score={self.score}"
