"""ORM models for experiment configurations, run metadata, and evaluation result snapshots."""

import uuid

from django.db import models


class Experiment(models.Model):
    STATUS_CHOICES = [
        ("pending", "pending"),
        ("running", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("stopped", "stopped"),
    ]
    STRATEGY_CHOICES = [
        ("bayesian", "bayesian"),
        ("grid", "grid"),
        ("random", "random"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="pending"
    )
    strategy = models.CharField(
        max_length=20, choices=STRATEGY_CHOICES, default="bayesian"
    )
    # {"chunk_size":[128,256,512,1024],"top_k":[3,5,10],"reranker":["none","cross_encoder"]}
    search_space = models.JSONField(default=dict)

    # ── Feature 1: Cost-Aware Optimization ────────────────────────────────────
    cost_budget_usd = models.DecimalField(
        max_digits=10, decimal_places=4, null=True, blank=True
    )
    cost_per_1k_input_tokens = models.DecimalField(
        max_digits=10, decimal_places=6, default=0.000150
    )
    cost_per_1k_output_tokens = models.DecimalField(
        max_digits=10, decimal_places=6, default=0.000600
    )
    total_cost_usd = models.DecimalField(
        max_digits=10, decimal_places=4, default=0
    )

    # ── Feature 2: Confidence-Based Stopping ──────────────────────────────────
    stopping_mode = models.CharField(
        max_length=20,
        choices=[("fixed", "fixed"), ("confidence", "confidence")],
        default="confidence",
    )
    max_iterations = models.PositiveIntegerField(default=30)
    ci_window = models.PositiveIntegerField(default=5)
    ci_threshold = models.FloatField(default=0.02)
    improvement_epsilon = models.FloatField(default=0.005)
    confidence_level = models.FloatField(default=0.95)

    # ── Feature 3: Meta-Optimizer ─────────────────────────────────────────────
    meta_optimizer_enabled = models.BooleanField(default=True)
    ucb_kappa = models.FloatField(default=1.96)
    variance_window = models.PositiveIntegerField(default=8)
    exploit_threshold = models.FloatField(default=0.001)

    # ── Feature 4: Naive Baseline ─────────────────────────────────────────────
    baseline_run_enabled = models.BooleanField(default=True)
    baseline_iteration_id = models.UUIDField(null=True, blank=True)

    celery_task_id = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.name} ({self.status})"


class IterationResult(models.Model):
    """
    Single table for every iteration in an experiment.
    iteration_number=0 is always the naive RAG baseline (is_baseline=True).
    """

    EXPLORATION_MODE_CHOICES = [
        ("explore", "explore"),
        ("exploit", "exploit"),
    ]
    STOPPING_REASON_CHOICES = [
        ("", "none"),
        ("ci_converged", "ci_converged"),
        ("no_improvement", "no_improvement"),
        ("budget_exhausted", "budget_exhausted"),
        ("max_iterations", "max_iterations"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="iterations"
    )
    iteration_number = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    config_snapshot = models.JSONField()
    is_baseline = models.BooleanField(default=False)

    # ── RAGAS metrics ──────────────────────────────────────────────────────────
    faithfulness = models.FloatField(null=True, blank=True)
    answer_relevancy = models.FloatField(null=True, blank=True)
    context_precision = models.FloatField(null=True, blank=True)
    context_recall = models.FloatField(null=True, blank=True)
    ragas_score = models.FloatField(null=True, blank=True)  # composite mean

    # ── IR metrics ────────────────────────────────────────────────────────────
    mrr = models.FloatField(null=True, blank=True)
    ndcg_at_k = models.FloatField(null=True, blank=True)
    precision_at_k = models.FloatField(null=True, blank=True)
    recall_at_k = models.FloatField(null=True, blank=True)

    # ── Feature 1: Cost per iteration ─────────────────────────────────────────
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    iteration_cost_usd = models.DecimalField(
        max_digits=10, decimal_places=6, default=0
    )
    cumulative_cost_usd = models.DecimalField(
        max_digits=10, decimal_places=6, default=0
    )

    # ── Feature 2: Stopping diagnostics ───────────────────────────────────────
    ci_lower = models.FloatField(null=True, blank=True)
    ci_upper = models.FloatField(null=True, blank=True)
    ci_width = models.FloatField(null=True, blank=True)
    mean_improvement = models.FloatField(null=True, blank=True)
    stopping_triggered = models.BooleanField(default=False)
    stopping_reason = models.CharField(max_length=50, blank=True, default="")

    # ── Feature 3: Meta-optimizer snapshot ────────────────────────────────────
    exploration_mode = models.CharField(
        max_length=10,
        choices=EXPLORATION_MODE_CHOICES,
        default="explore",
    )
    rolling_variance = models.FloatField(null=True, blank=True)
    ucb_score = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = [("experiment", "iteration_number")]
        ordering = ["iteration_number"]

    def __str__(self):
        tag = " [baseline]" if self.is_baseline else ""
        score = f"{self.ragas_score:.3f}" if self.ragas_score is not None else "—"
        return f"Iter {self.iteration_number}{tag}: ragas={score}"
