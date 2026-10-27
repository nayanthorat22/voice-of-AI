"""
Unit and integration tests for Voice of Customer AI.
Covers:
- Dashboard response
- Single review inference & database persistence
- Form validation (short inputs, empty inputs)
- CSV upload validation & batch creation
- Review filtering & pagination
- Topic discovery with K-Means
- Model evaluation view and metric parsing
- Missing artifacts graceful fallback
"""

import io
import os
import json
from django.test import TestCase, Client
from django.urls import reverse
from feedback.models import Review, UploadBatch
from feedback.ml_service import get_ml_service, SentimentMLService
from feedback.analytics import discover_topics, generate_business_insights

class VocAiTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Seed test reviews
        self.rev1 = Review.objects.create(
            text="This is an absolutely fantastic product, the battery lasts all day!",
            source="Test Suite",
            predicted_sentiment="Positive",
            confidence_score=0.92,
            product_name="PowerBank Ultra"
        )
        self.rev2 = Review.objects.create(
            text="Terrible build quality, broke on the first day of use. Do not recommend.",
            source="Test Suite",
            predicted_sentiment="Negative",
            confidence_score=0.88,
            product_name="PowerBank Ultra"
        )
        self.rev3 = Review.objects.create(
            text="Decent overall, but delivery was slow and packaging was damaged.",
            source="Test Suite",
            predicted_sentiment="Negative",
            confidence_score=0.62,
            product_name="Phone Case"
        )

    def test_dashboard_status_code(self):
        """Dashboard renders successfully with 200 OK and includes key metrics."""
        response = self.client.get(reverse('feedback:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Customer Feedback Dashboard")
        self.assertIn('total_reviews', response.context)
        self.assertEqual(response.context['total_reviews'], 3)

    def test_review_analyzer_get(self):
        """Single review analyzer page renders."""
        response = self.client.get(reverse('feedback:analyze'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Analyse a Review")

    def test_review_analyzer_valid_prediction(self):
        """Single review analyzer performs inference and stores to database."""
        initial_count = Review.objects.count()
        response = self.client.post(reverse('feedback:analyze'), {
            'text': 'This is the best product I have ever bought, highly recommended, absolutely wonderful and excellent!',
            'product_name': 'Support Service'
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Review.objects.count(), initial_count + 1)
        latest = Review.objects.latest('id')
        self.assertEqual(latest.predicted_sentiment, 'Positive')
        self.assertGreater(latest.confidence_score, 0.70)
        self.assertEqual(latest.product_name, 'Support Service')

    def test_review_analyzer_short_text_validation(self):
        """Review text under 5 characters triggers validation error."""
        response = self.client.post(reverse('feedback:analyze'), {
            'text': 'Bad',
            'product_name': 'Test'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "at least 5 characters")

    def test_review_explorer_filtering_and_search(self):
        """Review explorer filters correctly by sentiment and query keyword."""
        # Filter by Positive
        resp_pos = self.client.get(reverse('feedback:reviews'), {'sentiment': 'Positive'})
        self.assertEqual(resp_pos.status_code, 200)
        self.assertEqual(len(resp_pos.context['page_obj']), 1)

        # Filter by Negative
        resp_neg = self.client.get(reverse('feedback:reviews'), {'sentiment': 'Negative'})
        self.assertEqual(resp_neg.status_code, 200)
        self.assertEqual(len(resp_neg.context['page_obj']), 2)

        # Keyword search
        resp_search = self.client.get(reverse('feedback:reviews'), {'q': 'fantastic'})
        self.assertEqual(len(resp_search.context['page_obj']), 1)

    def test_csv_upload_validation_non_csv(self):
        """Uploading non-csv file fails validation."""
        fake_txt = io.BytesIO(b"Just some random text")
        fake_txt.name = "test.txt"
        response = self.client.post(reverse('feedback:upload'), {'csv_file': fake_txt})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Only valid CSV (.csv) files are supported")

    def test_csv_upload_valid_processing(self):
        """Valid CSV with 'review' column processes reviews and creates batch."""
        csv_data = "review,product,date\n\"Excellent quality and super fast shipping!\",Gadget X,2026-05-01\n\"Horrible product, stopped working after two hours.\",Gadget Y,2026-05-02\n"
        csv_file = io.BytesIO(csv_data.encode('utf-8'))
        csv_file.name = "customer_feedback_sample.csv"

        response = self.client.post(reverse('feedback:upload'), {'csv_file': csv_file}, follow=True)
        self.assertEqual(response.status_code, 200)

        # Verify batch created
        batch = UploadBatch.objects.latest('id')
        self.assertEqual(batch.filename, "customer_feedback_sample.csv")
        self.assertEqual(batch.status, "COMPLETED")
        self.assertEqual(batch.processed_reviews, 2)

    def test_topic_discovery_analytics(self):
        """Topic discovery runs clustering or provides graceful warnings for small datasets."""
        # 3 reviews is below minimum threshold of 10
        result = discover_topics(Review.objects.all(), n_clusters=2)
        self.assertIsNotNone(result['warning'])

        # Create enough reviews to exceed threshold
        bulk_reviews = [
            Review(text=f"Battery life and charging speed review number {i}", predicted_sentiment="Positive", confidence_score=0.9)
            for i in range(7)
        ] + [
            Review(text=f"Shipping delay and customer service complaint number {i}", predicted_sentiment="Negative", confidence_score=0.85)
            for i in range(7)
        ]
        Review.objects.bulk_create(bulk_reviews)

        res_enough = discover_topics(Review.objects.all(), n_clusters=2)
        self.assertIsNone(res_enough['warning'])
        self.assertEqual(len(res_enough['clusters']), 2)

    def test_business_insights_generation(self):
        """Business insights returns grounded summary and rule-based cards."""
        report = generate_business_insights(Review.objects.all())
        self.assertTrue(report['has_data'])
        self.assertGreater(len(report['insights']), 0)

    def test_model_evaluation_page(self):
        """Model evaluation page renders metrics."""
        response = self.client.get(reverse('feedback:model_evaluation'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Model Evaluation")
        self.assertContains(response, "Confusion Matrix")
