import os
import json
import csv
import io
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q, Avg, Count
from django.conf import settings
from .models import Review, UploadBatch
from .forms import ReviewAnalysisForm, CSVUploadForm
from .ml_service import get_ml_service, ModelNotLoadedError
from .analytics import (
    discover_topics,
    build_sentiment_donut_chart,
    build_timeline_chart,
    build_product_comparison_chart,
    generate_business_insights
)

METRICS_PATH = os.path.join(settings.BASE_DIR, 'artifacts', 'metrics.json')

def dashboard(request):
    """
    Main business dashboard: KPIs, distribution charts, latest reviews, and top topics.
    """
    total_reviews = Review.objects.count()
    pos_count = Review.objects.filter(predicted_sentiment='Positive').count()
    neg_count = Review.objects.filter(predicted_sentiment='Negative').count()
    neu_count = Review.objects.filter(predicted_sentiment='Neutral').count()

    pos_pct = round((pos_count / total_reviews * 100), 1) if total_reviews > 0 else 0
    neg_pct = round((neg_count / total_reviews * 100), 1) if total_reviews > 0 else 0
    neu_pct = round((neu_count / total_reviews * 100), 1) if total_reviews > 0 else 0

    # Load real model evaluation metrics if available
    metrics = None
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    # Plotly donut chart
    donut_chart_html = build_sentiment_donut_chart(pos_count, neg_count, neu_count)

    # Plotly product and timeline charts
    all_reviews_qs = Review.objects.all()
    timeline_chart_html = build_timeline_chart(all_reviews_qs)
    product_chart_html = build_product_comparison_chart(all_reviews_qs)

    # Recent 6 reviews
    latest_reviews = Review.objects.select_related('batch')[:6]

    # Quick topic preview (top 3)
    topics_data = discover_topics(Review.objects.all()[:400], n_clusters=3, top_terms=4)

    # Brief business insights preview
    insights_data = generate_business_insights(all_reviews_qs, topics_data)
    brief_insights = insights_data.get('insights', [])[:3]

    context = {
        'total_reviews': total_reviews,
        'pos_count': pos_count,
        'neg_count': neg_count,
        'neu_count': neu_count,
        'pos_pct': pos_pct,
        'neg_pct': neg_pct,
        'neu_pct': neu_pct,
        'metrics': metrics,
        'donut_chart_html': donut_chart_html,
        'timeline_chart_html': timeline_chart_html,
        'product_chart_html': product_chart_html,
        'latest_reviews': latest_reviews,
        'topics_preview': topics_data.get('clusters', []),
        'brief_insights': brief_insights,
    }
    return render(request, 'feedback/dashboard.html', context)


def upload_reviews(request):
    """
    Accepts CSV uploads, validates schema, classifies sentiment in batches, and persists records.
    """
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            filename = csv_file.name

            batch = UploadBatch.objects.create(
                filename=filename,
                status='PROCESSING'
            )

            try:
                # Read CSV into pandas DataFrame safely
                content = csv_file.read().decode('utf-8', errors='replace')
                df = pd.read_csv(io.StringIO(content))

                if df.empty:
                    batch.status = 'FAILED'
                    batch.error_message = "The uploaded CSV file is empty."
                    batch.save()
                    messages.error(request, "Uploaded file contains no rows.")
                    return redirect('feedback:upload')

                # Identify text column
                text_col = None
                for col in ['review', 'text', 'content', 'review_text', 'Review', 'Text', 'feedback']:
                    if col in df.columns:
                        text_col = col
                        break

                if not text_col:
                    batch.status = 'FAILED'
                    batch.error_message = (
                        f"Missing required text column. Uploaded columns: {list(df.columns)}. "
                        "Expected one of: 'review', 'text', 'content', 'review_text'."
                    )
                    batch.save()
                    messages.error(request, batch.error_message)
                    return redirect('feedback:upload')

                # Identify optional columns
                product_col = None
                for col in ['product', 'product_name', 'item', 'Product', 'Product Name']:
                    if col in df.columns:
                        product_col = col
                        break

                date_col = None
                for col in ['date', 'review_date', 'Date', 'timestamp']:
                    if col in df.columns:
                        date_col = col
                        break

                true_sentiment_col = None
                for col in ['sentiment', 'label', 'true_sentiment', 'Sentiment']:
                    if col in df.columns and col != text_col:
                        true_sentiment_col = col
                        break

                # Filter valid text rows
                df = df.dropna(subset=[text_col])
                df[text_col] = df[text_col].astype(str).str.strip()
                df = df[df[text_col].str.len() >= 3]

                batch.total_rows = len(df)
                if len(df) == 0:
                    batch.status = 'FAILED'
                    batch.error_message = "No non-empty review text rows found in file."
                    batch.save()
                    messages.error(request, "File contains no valid review text rows.")
                    return redirect('feedback:upload')

                # Classify sentiments using ML service
                ml_service = get_ml_service()
                texts = df[text_col].tolist()
                predictions = ml_service.predict_batch(texts)

                # Prepare Django Review model instances in chunks of 500
                reviews_to_create = []
                for idx, (_, row) in enumerate(df.iterrows()):
                    pred_data = predictions[idx]

                    # Parse product name
                    p_name = None
                    if product_col and pd.notna(row[product_col]):
                        p_name = str(row[product_col]).strip()[:255]

                    # Parse review date safely
                    r_date = None
                    if date_col and pd.notna(row[date_col]):
                        try:
                            r_date = pd.to_datetime(row[date_col]).date()
                        except Exception:
                            r_date = None

                    # Parse true sentiment label
                    t_sent = None
                    if true_sentiment_col and pd.notna(row[true_sentiment_col]):
                        raw_label = str(row[true_sentiment_col]).strip().capitalize()
                        if 'Pos' in raw_label or raw_label == '1':
                            t_sent = 'Positive'
                        elif 'Neg' in raw_label or raw_label == '0':
                            t_sent = 'Negative'

                    review_obj = Review(
                        text=row[text_col],
                        source=f"CSV: {filename[:40]}",
                        predicted_sentiment=pred_data['sentiment'],
                        confidence_score=pred_data['confidence'],
                        true_sentiment=t_sent,
                        product_name=p_name,
                        review_date=r_date,
                        batch=batch
                    )
                    reviews_to_create.append(review_obj)

                Review.objects.bulk_create(reviews_to_create, batch_size=500)

                batch.processed_reviews = len(reviews_to_create)
                batch.status = 'COMPLETED'
                batch.save()

                messages.success(
                    request,
                    f"Successfully processed and imported {batch.processed_reviews} reviews from '{filename}'."
                )
                return redirect('feedback:reviews')

            except ModelNotLoadedError as mle:
                batch.status = 'FAILED'
                batch.error_message = str(mle)
                batch.save()
                messages.error(request, str(mle))
                return redirect('feedback:upload')
            except Exception as e:
                batch.status = 'FAILED'
                batch.error_message = f"Processing exception: {str(e)}"
                batch.save()
                messages.error(request, f"Failed to process CSV: {str(e)}")
                return redirect('feedback:upload')
    else:
        form = CSVUploadForm()

    recent_batches = UploadBatch.objects.all()[:8]
    return render(request, 'feedback/upload.html', {'form': form, 'recent_batches': recent_batches})


def review_explorer(request):
    """
    Searchable, filterable, and paginated review explorer page.
    """
    query = request.GET.get('q', '').strip()
    sentiment_filter = request.GET.get('sentiment', '').strip()
    batch_filter = request.GET.get('batch', '').strip()
    product_filter = request.GET.get('product', '').strip()

    reviews_qs = Review.objects.select_related('batch').all()

    if query:
        reviews_qs = reviews_qs.filter(Q(text__icontains=query) | Q(product_name__icontains=query))

    if sentiment_filter in ['Positive', 'Negative', 'Neutral']:
        reviews_qs = reviews_qs.filter(predicted_sentiment=sentiment_filter)

    if batch_filter and batch_filter.isdigit():
        reviews_qs = reviews_qs.filter(batch_id=int(batch_filter))

    if product_filter:
        reviews_qs = reviews_qs.filter(product_name=product_filter)

    total_filtered = reviews_qs.count()

    # Pagination: 15 reviews per page
    paginator = Paginator(reviews_qs, 15)
    page_number = request.GET.get('page', 1)

    try:
        page_obj = paginator.get_page(page_number)
    except (PageNotAnInteger, EmptyPage):
        page_obj = paginator.get_page(1)

    # Context filters
    available_batches = UploadBatch.objects.filter(processed_reviews__gt=0)[:10]
    available_products = Review.objects.exclude(product_name__isnull=True).exclude(product_name='').values_list('product_name', flat=True).distinct()[:20]

    context = {
        'page_obj': page_obj,
        'total_filtered': total_filtered,
        'query': query,
        'sentiment_filter': sentiment_filter,
        'batch_filter': batch_filter,
        'product_filter': product_filter,
        'available_batches': available_batches,
        'available_products': available_products,
    }
    return render(request, 'feedback/reviews.html', context)


def review_analyzer(request):
    """
    Single review real-time analyzer with confidence meter and database persistence.
    """
    result = None
    if request.method == 'POST':
        form = ReviewAnalysisForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            product_name = form.cleaned_data.get('product_name', '').strip() or None

            try:
                ml_service = get_ml_service()
                prediction = ml_service.predict_one(text)

                # Persist to SQLite
                saved_review = Review.objects.create(
                    text=text,
                    source='Single Analysis',
                    predicted_sentiment=prediction['sentiment'],
                    confidence_score=prediction['confidence'],
                    product_name=product_name
                )

                result = {
                    'text': text,
                    'sentiment': prediction['sentiment'],
                    'confidence_percent': round(prediction['confidence'] * 100, 1),
                    'probabilities': prediction['probabilities'],
                    'is_low_confidence': prediction['is_low_confidence'],
                    'product_name': product_name,
                    'review_id': saved_review.id
                }
                messages.success(request, f"Review classified as {prediction['sentiment']}.")
            except ModelNotLoadedError as mle:
                messages.error(request, str(mle))
            except Exception as e:
                messages.error(request, f"Analysis error: {str(e)}")
    else:
        form = ReviewAnalysisForm()

    # Recent single analyses
    recent_single = Review.objects.filter(source='Single Analysis')[:5]

    return render(request, 'feedback/review_analyzer.html', {
        'form': form,
        'result': result,
        'recent_single': recent_single
    })


def topics_view(request):
    """
    Topic discovery using TF-IDF and K-Means clustering with configurable k.
    """
    try:
        k = int(request.GET.get('k', 4))
        k = max(2, min(k, 8))  # clamp between 2 and 8
    except (ValueError, TypeError):
        k = 4

    reviews_qs = Review.objects.all()
    topics_result = discover_topics(reviews_qs, n_clusters=k, top_terms=6)

    context = {
        'k': k,
        'clusters': topics_result['clusters'],
        'warning': topics_result['warning'],
        'total_reviews': topics_result['total_reviews']
    }
    return render(request, 'feedback/topics.html', context)


def model_evaluation(request):
    """
    Displays authentic model evaluation metrics loaded from artifacts/metrics.json.
    """
    metrics = None
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception as e:
            metrics = {'error': f"Failed to read metrics file: {str(e)}"}

    return render(request, 'feedback/model_evaluation.html', {'metrics': metrics})


def business_insights(request):
    """
    Generates actionable business suggestions based on analyzed reviews and topics.
    """
    reviews_qs = Review.objects.all()
    topics_data = discover_topics(reviews_qs[:500], n_clusters=4, top_terms=5)
    report = generate_business_insights(reviews_qs, topics_data)

    return render(request, 'feedback/insights.html', {
        'report': report,
        'topics': topics_data.get('clusters', [])
    })
