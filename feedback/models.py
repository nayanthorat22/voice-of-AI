from django.db import models
from django.utils import timezone

class UploadBatch(models.Model):
    """
    Tracks CSV file uploads and analysis batches.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    filename = models.CharField(max_length=255, help_text="Original uploaded filename")
    total_rows = models.IntegerField(default=0, help_text="Total rows in the uploaded file")
    processed_reviews = models.IntegerField(default=0, help_text="Successfully processed reviews count")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(blank=True, null=True, help_text="Error details if processing failed")
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Upload Batch'
        verbose_name_plural = 'Upload Batches'

    def __str__(self):
        return f"Batch #{self.id} - {self.filename} ({self.status})"


class Review(models.Model):
    """
    Stores individual customer reviews with predicted sentiment and metadata.
    """
    SENTIMENT_CHOICES = [
        ('Positive', 'Positive'),
        ('Negative', 'Negative'),
        ('Neutral', 'Neutral'),
    ]

    text = models.TextField(help_text="Customer feedback text")
    source = models.CharField(max_length=100, default='Direct Input', help_text="Source of the review, e.g. CSV upload or Single Analysis")
    predicted_sentiment = models.CharField(max_length=20, choices=SENTIMENT_CHOICES)
    confidence_score = models.FloatField(help_text="Model confidence score between 0.0 and 1.0")
    true_sentiment = models.CharField(max_length=20, blank=True, null=True, help_text="Ground truth label if available in source data")
    product_name = models.CharField(max_length=255, blank=True, null=True, help_text="Product or service name")
    review_date = models.DateField(blank=True, null=True, help_text="Original review date if present")
    batch = models.ForeignKey(UploadBatch, on_delete=models.SET_NULL, null=True, blank=True, related_name='reviews')
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['predicted_sentiment']),
            models.Index(fields=['product_name']),
            models.Index(fields=['created_at']),
        ]
        verbose_name = 'Review'
        verbose_name_plural = 'Reviews'

    def __str__(self):
        return f"[{self.predicted_sentiment} ({self.confidence_score*100:.1f}%)] {self.text[:50]}..."

    @property
    def confidence_percent(self):
        return round(self.confidence_score * 100, 1)

    @property
    def is_low_confidence(self):
        # A prediction close to 50% (under 65%) indicates high uncertainty
        return self.confidence_score < 0.65
