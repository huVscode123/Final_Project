# analyzer/urls.py
from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('',                                    views.dashboard,          name='dashboard'),
    path('upload/',                             views.upload_pcap,        name='upload'),
    path('sessions/',                           views.session_list,       name='sessions'),
    path('sessions/<int:pk>/',                  views.session_detail,     name='session_detail'),
    path('sessions/<int:pk>/status/',           views.session_status_api, name='session_status'),
    path('sessions/<int:pk>/gradcam/',          views.trigger_gradcam,    name='trigger_gradcam'),
    path('sessions/<int:pk>/gradcam/gallery/',  views.gradcam_gallery,    name='gradcam_gallery'),
    path('sessions/<int:pk>/delete/',           views.session_delete,     name='session_delete'),
    path('live/',                               views.live_monitor,       name='live'),
    path('ai-chat/',                            views.ai_chat,            name='ai_chat'),
]
