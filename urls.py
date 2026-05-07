from django.urls import path
from . import views

urlpatterns = [
    path('app/', views.index, name='app'),  # Air app
    path('video/', views.video_feed, name='video_feed'),
    path('set/<str:mode>/', views.set_mode, name='set_mode'),
    path('size/<str:action>/', views.change_size, name='size'),
]
