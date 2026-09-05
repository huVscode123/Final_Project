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
    path('sessions/<int:pk>/heatmap/',          views.heatmap_analysis,   name='heatmap_analysis'),
    path('sessions/<int:pk>/delete/',           views.session_delete,     name='session_delete'),
    path('simulation/',                         views.simulation,         name='simulation'),
    path('simulation/api/',                     views.simulation_api,     name='simulation_api'),
    path('simulation/download/<str:filename>/', views.simulation_pcap_download,
         name='simulation_pcap_download'),
    path('ablation/',                           views.ablation_dashboard, name='ablation'),
    path('ablation/api/run/',                   views.ablation_run_api,   name='ablation_run'),
    path('ablation/api/status/',                views.ablation_status_api,name='ablation_status'),
    path('live/',                               views.live_monitor,       name='live'),
    path('ai-chat/',                            views.ai_chat,            name='ai_chat'),
]
