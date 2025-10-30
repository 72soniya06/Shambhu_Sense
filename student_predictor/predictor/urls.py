# predictor/urls.py
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('landing/', views.landing_page, name='landing'),
    path('role_select/', views.role_select, name='role_select'),

    path('login/', views.login_view, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),

    # student pages
    path('predict/manual/', views.predict_manual, name='predict_manual'),
    path('predict/csv/', views.predict_csv, name='predict_csv'),
    path('download/<str:filename>/', views.download_file, name='download_file'),
    path("chatbot/", views.chatbot_page, name="chatbot"),  # 👈 Chatbot UI
    path("chatbot/api/", views.chatbot_api, name="chatbot_api"),  # 👈 Backend AI API
    path("chatbot/history/", views.chat_history, name="chat_history"),  # 👈 Chat History
    path("chatbot/delete/<int:chat_id>/", views.delete_chat, name="delete_chat"),  # 👈 Delete chat
    path("my-library/", views.my_library, name="my_library"),
    path('add-to-library/<str:title>/<path:file_url>/', views.add_to_library, name='add_to_library'),
    path('delete-from-library/<int:file_id>/', views.delete_from_library, name='delete_from_library'),
    path('search/', views.search_papers, name='search_papers'),


    # teacher page
    path('train/', views.train_view, name='train'),

    path('set-password/', views.set_password_view, name='set_password'),

    # 🔹 New Profile URLs
    path('profile/', views.profile_view, name='profile_view'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
   ]

