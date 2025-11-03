from django.urls import path 
from api.views import liveness_view

urlpatterns = [
    path('liveness-spoofing-score', liveness_view.liveness_spoofing_score, name='get-liveness-spoofing-score')
]
