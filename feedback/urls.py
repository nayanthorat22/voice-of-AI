from django.urls import path
from . import views

app_name = 'feedback'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_reviews, name='upload'),
    path('reviews/', views.review_explorer, name='reviews'),
    path('analyze/', views.review_analyzer, name='analyze'),
    path('topics/', views.topics_view, name='topics'),
    path('model-evaluation/', views.model_evaluation, name='model_evaluation'),
    path('insights/', views.business_insights, name='insights'),
]
