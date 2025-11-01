# predictor/urls.py
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('landing/', views.landing_page, name='landing'),
    path('role_select/', views.role_select, name='role_select'),

    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),

    # Student pages
    path('predict/manual/', views.predict_manual, name='predict_manual'),
    path('predict/csv/', views.predict_csv, name='predict_csv'),

    # Downloads
    #path('download/<str:filename>/', views.download_file, name='download_file'),

    # Chatbot
    path("chatbot/", views.chatbot_page, name="chatbot"),
    path("chatbot/api/", views.chatbot_api, name="chatbot_api"),
    path("chatbot/history/", views.chat_history, name="chat_history"),
    path("chatbot/delete/<int:chat_id>/", views.delete_chat, name="delete_chat"),

    # Teacher & Profile pages
    path('train/', views.train_view, name='train'),
    path('set-password/', views.set_password_view, name='set_password'),
    path('profile/', views.profile_view, name='profile_view'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
]
