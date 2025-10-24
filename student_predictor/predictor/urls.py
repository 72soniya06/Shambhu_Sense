# predictor/urls.py
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.role_select, name='role_select'),
    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),

    # student pages
    path('predict/manual/', views.predict_manual, name='predict_manual'),
    path('predict/csv/', views.predict_csv, name='predict_csv'),
    path('download/<str:filename>/', views.download_file, name='download_file'),

    # teacher page
    path('train/', views.train_view, name='train'),

    path('set-password/', views.set_password_view, name='set_password'),

    # 🔹 New Profile URLs
    path('profile/', views.profile_view, name='profile_view'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
]

