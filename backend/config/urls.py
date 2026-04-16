"""Root URL configuration that wires together all app-level URL routers."""

from django.urls import include, path


urlpatterns = [
    path("api/experiments/", include("apps.experiments.urls", namespace="experiments")),
]
