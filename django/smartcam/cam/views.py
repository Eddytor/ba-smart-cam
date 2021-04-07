from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection
from django.http import JsonResponse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.shortcuts import render
from django import forms
from PIL import Image
from .forms import ImageForm
from django.core.files import File
from io import BytesIO
from .models import ImageFile
from .models import ImageView
from django.conf import settings
from django.contrib.auth.decorators import login_required
import os


def home(request):
    return HttpResponse('<h1>Smart Cam<h1>')


@login_required
def db_request(request):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM detections")
    dicts = dictfetchall(cursor)

    return JsonResponse(dicts, safe=False)


@login_required
def db_request_cams(request):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM cameras")
    dicts = dictfetchall(cursor)

    return JsonResponse(dicts, safe=False)


@login_required
def image_view(request):
    return render(request, 'cam/image.html', context={"view": ImageView.objects.first()})


def img_request(request):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM detections")
    dicts = dictfetchall(cursor)
    for entry in dicts:
        img = Image.open(entry['image_path'])
        print(entry['image_path'])
        head_tail = os.path.split(entry['image_path'])
        print("tail", head_tail[1])
        new_path = settings.MEDIA_ROOT + head_tail[1]
        print("new path", new_path)
        img.save(new_path, "JPEG")
    return HttpResponse('<h1>Updated Images<h1>')


def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

