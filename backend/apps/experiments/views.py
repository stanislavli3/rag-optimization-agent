"""API views for creating, listing, and comparing RAG optimization experiments."""

import json
import time

from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import AgentEvent, Experiment


SSE_POLL_INTERVAL = 0.5
SSE_IDLE_TIMEOUT = 60 * 30  # 30 minutes of no new events ends the stream.


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_experiment(request, experiment_id):
    """Server-Sent Events feed of AgentEvent rows for a single experiment.

    Emits one `data:` message per new AgentEvent row, polling every 500 ms.
    Closes when the Experiment reaches a terminal status (completed / failed /
    stopped) or after SSE_IDLE_TIMEOUT seconds with no new events — whichever
    comes first.
    """
    # Fail fast on unknown experiment rather than streaming an empty feed.
    get_object_or_404(Experiment, pk=experiment_id)

    def event_generator():
        last_seen = 0
        last_activity = time.monotonic()
        # Replay any events already written so a late-connecting client still
        # gets the full tree history. Live rows are appended below.
        while True:
            qs = AgentEvent.objects.filter(
                experiment_id=experiment_id, id__gt=last_seen
            ).order_by("id")
            batch = list(qs)
            if batch:
                last_activity = time.monotonic()
                for event in batch:
                    body = {
                        "event": event.event_type,
                        "data": event.payload,
                        "id": event.id,
                    }
                    yield _sse(body)
                    last_seen = event.id

            try:
                exp = Experiment.objects.only("status").get(pk=experiment_id)
            except Experiment.DoesNotExist:
                yield _sse({"event": "error", "data": {"reason": "experiment_deleted"}})
                return
            if exp.status in ("completed", "failed", "stopped"):
                # Drain any events that landed between the last poll and the
                # status flip so the client never misses the "complete" frame.
                tail = AgentEvent.objects.filter(
                    experiment_id=experiment_id, id__gt=last_seen
                ).order_by("id")
                for event in tail:
                    yield _sse(
                        {
                            "event": event.event_type,
                            "data": event.payload,
                            "id": event.id,
                        }
                    )
                    last_seen = event.id
                yield _sse({"event": "stream_end", "data": {"status": exp.status}})
                return

            if time.monotonic() - last_activity > SSE_IDLE_TIMEOUT:
                yield _sse({"event": "stream_end", "data": {"status": "idle_timeout"}})
                return

            time.sleep(SSE_POLL_INTERVAL)

    response = StreamingHttpResponse(
        event_generator(), content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@require_http_methods(["GET"])
def list_events(request, experiment_id):
    """Non-streaming snapshot of all events for an experiment (for clients that
    can't use EventSource, e.g. during tests)."""
    events = AgentEvent.objects.filter(experiment_id=experiment_id).order_by("id")
    return JsonResponse(
        {
            "events": [
                {
                    "id": e.id,
                    "event": e.event_type,
                    "data": e.payload,
                    "created_at": e.created_at.isoformat(),
                }
                for e in events
            ]
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def emit_event(request, experiment_id):
    """Ingestion endpoint used by the BFTS loop worker to push events.

    Body: {"event_type": "...", "payload": {...}}
    """
    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid_json"}, status=400)

    event_type = body.get("event_type")
    if not event_type:
        return JsonResponse({"error": "event_type required"}, status=400)

    try:
        exp = Experiment.objects.get(pk=experiment_id)
    except Experiment.DoesNotExist:
        raise Http404("experiment not found")

    ev = AgentEvent.objects.create(
        experiment=exp,
        event_type=event_type,
        payload=body.get("payload", {}),
    )
    return JsonResponse({"id": ev.id, "event_type": ev.event_type}, status=201)
