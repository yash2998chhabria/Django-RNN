from django.urls import path
from . import views

urlpatterns = [
    path('',views.displayform,name="displayform"),
    path('homepage/',views.checkhome),
    path('resultspage/',views.checkresults)
]
