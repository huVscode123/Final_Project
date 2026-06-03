# reports/urls.py
from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    path('export/<int:session_pk>/',   views.export_report,  name='export'),
    path('download/<int:pk>/',         views.download_report, name='download'),
    path('session/<int:session_pk>/',  views.report_list,    name='list'),
]
