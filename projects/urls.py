# projects/urls.py
from django.urls import path
from . import views

app_name = 'projects'

urlpatterns = [
    path('',                              views.project_list,        name='list'),
    path('create/',                       views.project_create,      name='create'),
    path('<int:pk>/',                     views.project_detail,      name='detail'),
    path('<int:pk>/edit/',                views.project_edit,        name='edit'),
    path('<int:pk>/archive/',             views.project_archive,     name='archive'),
    path('<int:pk>/members/add/',         views.member_add,          name='member_add'),
    path('<int:pk>/members/<int:user_pk>/remove/', views.member_remove, name='member_remove'),
    path('<int:pk>/upload/',              views.packet_file_upload,  name='upload'),
    path('<int:pk>/files/<int:file_pk>/delete/', views.packet_file_delete, name='file_delete'),
]
