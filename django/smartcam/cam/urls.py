from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='smartcam-home'),
    path('db/detections/', views.db_request, name='smartcam-db-detections'),
    path('db/cams/', views.db_request_cams, name='smartcam-db-cams'),
    path('img/', views.img_request, name='smartcam-img'),
]