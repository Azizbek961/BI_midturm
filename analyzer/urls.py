from django.urls import path

from .views import ai_chat_view, analyze_data_view, query_data_view, upload_file_view


urlpatterns = [
    path('', upload_file_view, name='upload'),
    path('dashboard/', analyze_data_view, name='dashboard'),
    path('query/', query_data_view, name='query'),
    path('ai-chat/', ai_chat_view, name='ai_chat'),
]
