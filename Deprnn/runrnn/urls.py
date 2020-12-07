from django.urls import path
from . import views

urlpatterns = [
    path('',views.displayform,name="displayform")
]
