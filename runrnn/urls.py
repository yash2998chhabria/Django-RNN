from django.urls import path
from . import views

urlpatterns = [
    path('eh',views.displayform,name="displayform"),
    path('',views.checkhome),
    path('resultspage/',views.checkresults),
    path('about/',views.aboutpage)
]
