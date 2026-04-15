"""ORM models for uploaded documents and their index metadata stored in the vector store."""

import uuid

from django.db import models


class Document(models.Model):
    STATUS_CHOICES = [
        ("pending", "pending"),
        ("indexed", "indexed"),
        ("failed", "failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=1024)
    file_size_bytes = models.PositiveIntegerField()
    mime_type = models.CharField(max_length=100)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    def __str__(self):
        return f"{self.filename} ({self.status})"


class IndexMetadata(models.Model):
    document = models.OneToOneField(
        Document, on_delete=models.CASCADE, related_name="index_meta"
    )
    chroma_collection_id = models.CharField(max_length=255)
    chunk_count = models.PositiveIntegerField()
    chunk_size = models.PositiveIntegerField()
    chunk_overlap = models.FloatField()
    embedding_model = models.CharField(max_length=100)
    indexed_at = models.DateTimeField(auto_now_add=True)
    index_version = models.PositiveIntegerField(default=1)

    def __str__(self):
        return f"Index for {self.document.filename} (v{self.index_version})"


class TestQuestion(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="test_questions",
        null=True,
        blank=True,
    )
    question = models.TextField()
    ground_truth = models.TextField()
    auto_generated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question[:80]
