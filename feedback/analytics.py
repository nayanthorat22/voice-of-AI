"""
feedback/analytics.py
Customer feedback analytics:
1. Topic discovery using TF-IDF and K-Means clustering
2. Plotly chart builders for sentiment distribution, time trends, product comparisons
3. Rule-based practical business insight generator
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from django.core.cache import cache
import plotly.graph_objects as go
import plotly.express as px

# Colors aligned with clean dashboard palette
COLOR_POS = '#27845A'
COLOR_NEG = '#C74747'
COLOR_NEU = '#777F89'
COLOR_PRIMARY = '#247A78'

def discover_topics(reviews_queryset, n_clusters=4, top_terms=6):
    """
    Performs K-Means clustering on review texts using TF-IDF.
    """
    reviews_list = list(reviews_queryset.values('id', 'text', 'predicted_sentiment', 'confidence_score'))
    total_reviews = len(reviews_list)

    if total_reviews < 10:
        return {
            'clusters': [],
            'warning': f"Topic analysis requires at least 10 reviews (currently {total_reviews} available). Upload reviews to view topics.",
            'n_clusters': n_clusters,
            'total_reviews': total_reviews
        }

    if n_clusters > total_reviews:
        n_clusters = max(2, total_reviews // 2)

    texts = [r['text'] for r in reviews_list]

    try:
        custom_stopwords = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
            'product', 'item', 'amazon', 'bought', 'buy', 'order', 'ordered', 'one', 'would', 'really',
            'got', 'use', 'using', 'get', 'like', 'good', 'great', 'just', 'time', 'day', 'days', 'even'
        ]

        vectorizer = TfidfVectorizer(
            stop_words=custom_stopwords,
            max_features=1500,
            min_df=2 if total_reviews > 20 else 1,
            max_df=0.85,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        if tfidf_matrix.shape[1] < n_clusters:
            return {
                'clusters': [],
                'warning': "The review texts have insufficient vocabulary variety to form meaningful clusters.",
                'n_clusters': n_clusters,
                'total_reviews': total_reviews
            }

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()

        clusters = []
        for i in range(n_clusters):
            top_words = [terms[ind] for ind in order_centroids[i, :top_terms]]
            cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
            cluster_count = len(cluster_indices)

            cluster_reviews = [reviews_list[idx] for idx in cluster_indices]
            pos_count = sum(1 for r in cluster_reviews if r['predicted_sentiment'] == 'Positive')
            neg_count = sum(1 for r in cluster_reviews if r['predicted_sentiment'] == 'Negative')

            pos_pct = round((pos_count / cluster_count * 100), 1) if cluster_count > 0 else 0
            neg_pct = round((neg_count / cluster_count * 100), 1) if cluster_count > 0 else 0

            sample_reviews = [
                {
                    'text': r['text'][:160] + ('...' if len(r['text']) > 160 else ''),
                    'sentiment': r['predicted_sentiment'],
                    'confidence': round(r['confidence_score'] * 100, 1)
                }
                for r in cluster_reviews[:3]
            ]

            topic_title = f"Topic {i+1}: " + ", ".join(top_words[:3]).title()

            clusters.append({
                'id': i,
                'title': topic_title,
                'keywords': top_words,
                'count': cluster_count,
                'percent': round((cluster_count / total_reviews * 100), 1),
                'positive_count': pos_count,
                'negative_count': neg_count,
                'positive_pct': pos_pct,
                'negative_pct': neg_pct,
                'sample_reviews': sample_reviews,
            })

        clusters.sort(key=lambda x: x['count'], reverse=True)

        return {
            'clusters': clusters,
            'warning': None,
            'n_clusters': n_clusters,
            'total_reviews': total_reviews
        }

    except Exception as e:
        return {
            'clusters': [],
            'warning': f"Clustering error: {str(e)}",
            'n_clusters': n_clusters,
            'total_reviews': total_reviews
        }


def build_sentiment_donut_chart(pos_count, neg_count, neu_count=0):
    """
    Builds a clean, non-distracting Plotly donut chart for sentiment distribution.
    """
    total = pos_count + neg_count + neu_count
    if total == 0:
        return None

    labels = []
    values = []
    colors = []

    if pos_count > 0:
        labels.append('Positive')
        values.append(pos_count)
        colors.append(COLOR_POS)
    if neg_count > 0:
        labels.append('Negative')
        values.append(neg_count)
        colors.append(COLOR_NEG)
    if neu_count > 0:
        labels.append('Neutral')
        values.append(neu_count)
        colors.append(COLOR_NEU)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=1.5)),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(size=12, color='#252B33')
    )])

    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5, font=dict(size=11, color='#68717D')),
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=240,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})


def build_timeline_chart(reviews_qs):
    """
    Builds a clean line chart if review dates are present.
    """
    dated_reviews = reviews_qs.exclude(review_date__isnull=True).values('review_date', 'predicted_sentiment')
    if not dated_reviews.exists():
        return None

    df = pd.DataFrame(list(dated_reviews))
    if df.empty or len(df) < 3:
        return None

    df['review_date'] = pd.to_datetime(df['review_date'])
    grouped = df.groupby([pd.Grouper(key='review_date', freq='W-MON'), 'predicted_sentiment']).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    if 'Positive' in grouped.columns:
        fig.add_trace(go.Scatter(
            x=grouped['review_date'], y=grouped['Positive'],
            mode='lines+markers', name='Positive',
            line=dict(color=COLOR_POS, width=2),
            marker=dict(size=5)
        ))
    if 'Negative' in grouped.columns:
        fig.add_trace(go.Scatter(
            x=grouped['review_date'], y=grouped['Negative'],
            mode='lines+markers', name='Negative',
            line=dict(color=COLOR_NEG, width=2),
            marker=dict(size=5)
        ))

    fig.update_layout(
        title=dict(text="Weekly Sentiment Trend", font=dict(size=13, color='#252B33')),
        xaxis=dict(showgrid=True, gridcolor='#ECEFF2', linecolor='#DDE2E7'),
        yaxis=dict(showgrid=True, gridcolor='#ECEFF2', linecolor='#DDE2E7'),
        margin=dict(t=35, b=30, l=35, r=15),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})


def build_product_comparison_chart(reviews_qs):
    """
    Builds a clean stacked bar chart comparing sentiment across products.
    """
    prod_reviews = reviews_qs.exclude(product_name__isnull=True).exclude(product_name__exact='').values('product_name', 'predicted_sentiment')
    if not prod_reviews.exists():
        return None

    df = pd.DataFrame(list(prod_reviews))
    if df.empty or df['product_name'].nunique() < 2:
        return None

    top_prods = df['product_name'].value_counts().head(8).index
    df_filtered = df[df['product_name'].isin(top_prods)]

    counts = df_filtered.groupby(['product_name', 'predicted_sentiment']).size().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    if 'Positive' in counts.columns:
        fig.add_trace(go.Bar(
            y=counts['product_name'], x=counts['Positive'],
            name='Positive', orientation='h',
            marker_color=COLOR_POS
        ))
    if 'Negative' in counts.columns:
        fig.add_trace(go.Bar(
            y=counts['product_name'], x=counts['Negative'],
            name='Negative', orientation='h',
            marker_color=COLOR_NEG
        ))

    fig.update_layout(
        barmode='stack',
        title=dict(text="Sentiment by Product", font=dict(size=13, color='#252B33')),
        xaxis=dict(showgrid=True, gridcolor='#ECEFF2'),
        yaxis=dict(showgrid=False),
        margin=dict(t=35, b=25, l=110, r=15),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})


def generate_business_insights(reviews_qs, topics_data=None):
    """
    Generates practical, grounded findings and suggestions.
    """
    total = reviews_qs.count()
    if total == 0:
        return {
            'has_data': False,
            'summary': "No reviews analyzed yet.",
            'insights': []
        }

    pos_count = reviews_qs.filter(predicted_sentiment='Positive').count()
    neg_count = reviews_qs.filter(predicted_sentiment='Negative').count()
    low_conf_count = reviews_qs.filter(confidence_score__lt=0.65).count()

    pos_pct = round((pos_count / total * 100), 1)
    neg_pct = round((neg_count / total * 100), 1)
    low_conf_pct = round((low_conf_count / total * 100), 1)

    insights = []

    # 1. Overall sentiment
    if neg_pct >= 40.0:
        insights.append({
            'type': 'danger',
            'badge': 'High Negative Share',
            'title': 'Elevated Negative Sentiment',
            'metric': f"{neg_pct}% Negative ({neg_count} of {total} reviews)",
            'finding': f"A large portion ({neg_pct}%) of reviews express dissatisfaction.",
            'recommendation': "Review recurring complaints in recent customer uploads to identify primary causes of dissatisfaction."
        })
    elif pos_pct >= 70.0:
        insights.append({
            'type': 'success',
            'badge': 'High Satisfaction',
            'title': 'Predominantly Positive Feedback',
            'metric': f"{pos_pct}% Positive ({pos_count} of {total} reviews)",
            'finding': f"Over two-thirds ({pos_pct}%) of recorded customer feedback is positive.",
            'recommendation': "Continue monitoring feedback to ensure consistency across product batches."
        })
    else:
        insights.append({
            'type': 'info',
            'badge': 'Mixed Feedback',
            'title': 'Balanced Positive and Negative Feedback',
            'metric': f"{pos_pct}% Positive vs {neg_pct}% Negative",
            'finding': "Feedback is divided between satisfied and unsatisfied customers.",
            'recommendation': "Segment reviews by product or date to pinpoint areas needing improvement."
        })

    # 2. Ambiguity / Model Confidence check
    if low_conf_pct >= 15.0:
        insights.append({
            'type': 'warning',
            'badge': 'Model Uncertainty',
            'title': 'Reviews with Low Prediction Confidence',
            'metric': f"{low_conf_pct}% Low Confidence ({low_conf_count} reviews with <65% confidence)",
            'finding': f"Around {low_conf_pct}% of reviews contain ambiguous or mixed language where model confidence was below 65%.",
            'recommendation': "Perform manual checks on low-confidence reviews to verify true sentiment."
        })

    # 3. Topic-driven insight
    if topics_data and topics_data.get('clusters'):
        clusters = topics_data['clusters']
        most_negative_cluster = max(clusters, key=lambda c: c['negative_pct'])
        if most_negative_cluster['negative_pct'] >= 45.0 and most_negative_cluster['count'] >= 3:
            kw_str = ", ".join(most_negative_cluster['keywords'][:3])
            insights.append({
                'type': 'danger',
                'badge': 'Frequent Complaint Topic',
                'title': f"High Dissatisfaction in {most_negative_cluster['title']}",
                'metric': f"{most_negative_cluster['negative_pct']}% Negative in this topic",
                'finding': f"Reviews mentioning [{kw_str}] have the highest proportion of negative ratings.",
                'recommendation': f"Investigate customer comments related to [{kw_str}] as a priority."
            })

        largest_cluster = clusters[0]
        largest_kw = ", ".join(largest_cluster['keywords'][:3])
        insights.append({
            'type': 'info',
            'badge': 'Highest Volume Topic',
            'title': f"Most Discussed Topic: {largest_cluster['title']}",
            'metric': f"{largest_cluster['percent']}% of all reviews ({largest_cluster['count']} reviews)",
            'finding': f"Customers most frequently talk about topics related to [{largest_kw}].",
            'recommendation': "Review this topic's sample feedback to understand customer expectations."
        })

    return {
        'has_data': True,
        'summary': f"Based on {total} total reviews ({pos_count} positive, {neg_count} negative).",
        'pos_pct': pos_pct,
        'neg_pct': neg_pct,
        'total': total,
        'insights': insights
    }
