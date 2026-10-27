from django.contrib import admin
from .models import Review, UploadBatch

@admin.register(UploadBatch)
class UploadBatchAdmin(admin.ModelAdmin):
    list_display = ('id', 'filename', 'processed_reviews', 'total_rows', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('filename', 'error_message')
    readonly_fields = ('created_at',)

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'short_text', 'predicted_sentiment', 'confidence_display', 'product_name', 'source', 'created_at')
    list_filter = ('predicted_sentiment', 'source', 'created_at')
    search_fields = ('text', 'product_name')
    readonly_fields = ('created_at',)

    def short_text(self, obj):
        return obj.text[:60] + '...' if len(obj.text) > 60 else obj.text
    short_text.short_description = 'Review Text'

    def confidence_display(self, obj):
        return f"{obj.confidence_score * 100:.1f}%"
    confidence_display.short_description = 'Confidence'
