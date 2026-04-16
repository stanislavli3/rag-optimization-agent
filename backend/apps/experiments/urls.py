"""URL patterns for the experiments app, including CRUD and comparison endpoints."""

from django.urls import path

from . import views


app_name = "experiments"

urlpatterns = [
    path("<uuid:experiment_id>/stream/", views.stream_experiment, name="stream"),
    path("<uuid:experiment_id>/events/", views.list_events, name="events"),
    path("<uuid:experiment_id>/events/emit/", views.emit_event, name="emit_event"),
]
