# Voice of Customer AI — Customer Feedback Intelligence Platform

A full-stack Machine Learning Web Application built with **Django 5.2 LTS**, **Scikit-learn**, **Pandas**, **Plotly**, and **Bootstrap 5**. Designed as an enterprise-grade portfolio project for an AI & Data Science undergraduate applying for software engineering and ML internships.

---

## 1. Executive Summary & Problem Solved

Modern businesses receive tens of thousands of customer reviews across e-commerce marketplaces, app stores, and helpdesk tickets. Manually reading these reviews is unscalable, leading to delayed bug discovery, high churn, and missed product enhancement opportunities.

**Voice of Customer AI** is a complete, locally running intelligence web platform that:
1. **Automates Sentiment Classification** using a genuine Scikit-learn Machine Learning pipeline (`TF-IDF + Logistic Regression`).
2. **Discovers Unsupervised Emergent Topics** across reviews using `TF-IDF + K-Means Clustering`.
3. **Presents Executive Analytics** with interactive Plotly donut charts, product comparative bars, and timeline trends.
4. **Delivers Grounded Business Recommendations** through rule-based decision support cards that connect sentiment metrics directly to actionable product and operational priorities.

---

## 2. Technology Stack & Architecture

- **Backend Framework:** Django 5.2 LTS (MVT Architecture)
- **Programming Language:** Python 3.11
- **Machine Learning & NLP:** Scikit-learn 1.9, Pandas 3.1, NumPy 2.4, Joblib 1.6
- **Clustering:** K-Means with configurable cluster count ($k=2 \dots 8$)
- **Data Ingestion:** Hugging Face Datasets (`fancyzhx/amazon_polarity`)
- **Data Visualizations:** Plotly.js 2.30 responsive figures embedded in Django templates
- **Frontend / UI:** HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons, Custom CSS
- **Database:** SQLite with index optimization on sentiment and products
- **Execution Platform:** 100% Local on Windows PowerShell (No GPU or paid cloud API required)

### Architecture & Request Flow

```
[Customer Reviews CSV / Text Input]
                 │
                 ▼
       Django Forms & Views (Validation & CSRF Check)
                 │
                 ├──► SentimentMLService (Joblib Pipeline: TF-IDF + Logistic Regression)
                 │         │
                 │         ▼
                 ├──► SQLite DB (Review & UploadBatch ORM Models)
                 │         │
                 ▼         ▼
      Analytics Engine (K-Means Topic Modeling + Plotly + Business Rule Engine)
                 │
                 ▼
      Django Responsive Templates (Dashboard, Explorer, Topics, Insights, Eval)
```

---

## 3. Real Dataset & Model Benchmark Metrics

The model was trained on **12,000 authentic, labeled Amazon customer reviews** streamed via Hugging Face (`fancyzhx/amazon_polarity`), balanced 50% positive and 50% negative.

### Training & Evaluation Methodology:
- **Split:** 80% Training ($N=9,600$), 20% Held-Out Testing ($N=2,400$) with stratified sampling (`random_state=42`).
- **Feature Extraction:** Sublinear TF-IDF with unigrams and bigrams (`ngram_range=(1,2)`), `min_df=3`, `max_features=25000`.
- **Classification:** L2-regularized Logistic Regression ($C=1.5$, L-BFGS solver).
- **Artifacts Saved:** `artifacts/sentiment_model.joblib` and `artifacts/metrics.json`.

### Actual Calculated Benchmark Metrics:

| Metric | Score | Support |
| :--- | :--- | :--- |
| **Accuracy** | **88.08%** | 2,400 test samples |
| **Macro F1-Score** | **0.8808** | 2,400 test samples |
| **Macro Precision** | **0.8809** | 2,400 test samples |
| **Macro Recall** | **0.8808** | 2,400 test samples |
| **Negative Class F1** | **0.8812** | 1,200 test samples |
| **Positive Class F1** | **0.8804** | 1,200 test samples |

### Confusion Matrix on Held-Out Test Set:
- **True Negative (TN):** 1,061
- **False Positive (FP):** 139
- **False Negative (FN):** 147
- **True Positive (TP):** 1,053

---

## 4. Key Application Features

1. **Executive Dashboard (`/`):**
   - KPI cards for Total Reviews, Positive Ratio, Negative Ratio, and Model F1-Score.
   - Live Plotly sentiment distribution donut chart.
   - Dynamic preview of top discussion themes with sentiment ratios.
   - Stream of latest classified feedback with confidence indicators.

2. **Batch CSV Ingestion (`/upload/`):**
   - Accepts CSV files up to 10MB with columns (`review`, `text`, `content`, or `review_text`).
   - Supports optional product and review date columns.
   - Vectorized batch prediction with `Review.objects.bulk_create` for high performance.
   - Tracks upload batches, row counts, and error states.

3. **Customer Review Explorer (`/reviews/`):**
   - Full-text search and keyword query.
   - Filter by sentiment (Positive / Negative) and product SKU.
   - 15-item pagination with low-confidence warning badges.

4. **Single Review Analyzer (`/analyze/`):**
   - Instant sentiment classification for single customer comments.
   - Dual-class probability meters and low-confidence warning alert when confidence is $<65\%$.
   - Automatically saves and catalogs analyzed reviews in SQLite.

5. **Unsupervised Topic Discovery (`/topics/`):**
   - Real-time TF-IDF vectorization and K-Means clustering.
   - User-selectable cluster count ($k=2 \dots 8$).
   - Extraction of characteristic n-gram keywords per cluster.
   - Intra-cluster sentiment breakdown and exemplar customer quotes.

6. **Actionable Business Intelligence Report (`/insights/`):**
   - Rule-based decision-support cards (Urgent Action, Positive Health, Ambiguity Warning, Topic Focus).
   - Direct linkages between empirical metrics and suggested business interventions.
   - Print/PDF export support.

7. **Model Evaluation & Transparent Audit (`/model-evaluation/`):**
   - Displays real metrics, hyperparameters, confusion matrix, and per-class reports loaded from `artifacts/metrics.json`.

---

## 5. Local Setup & Execution (Windows PowerShell)

Open **PowerShell** in the project directory:

```powershell
cd D:\voice
```

### Initial One-Time Setup:
```powershell
# 1. Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# 2. Upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Install required dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 4. Stream real Amazon review dataset (12,000 samples)
.\.venv\Scripts\python.exe prepare_data.py

# 5. Train and evaluate the ML pipeline (saves artifacts)
.\.venv\Scripts\python.exe train_model.py

# 6. Apply database schema migrations
.\.venv\Scripts\python.exe manage.py makemigrations
.\.venv\Scripts\python.exe manage.py migrate

# 7. (Optional) Create admin user for Django Admin interface
.\.venv\Scripts\python.exe manage.py createsuperuser
```

### Starting the Server (Daily Usage):
```powershell
.\.venv\Scripts\python.exe manage.py runserver
```

Open your browser at **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**.

---

## 6. Running Automated Tests

Run the comprehensive 10-test test suite:

```powershell
.\.venv\Scripts\python.exe manage.py test
```

### Test Coverage:
- `test_dashboard_status_code`: Verifies HTTP 200 response and metric context.
- `test_review_analyzer_get`: Verifies form rendering.
- `test_review_analyzer_valid_prediction`: Tests ML inference and SQLite persistence.
- `test_review_analyzer_short_text_validation`: Verifies input length constraints.
- `test_review_explorer_filtering_and_search`: Tests sentiment filtering and keyword search.
- `test_csv_upload_validation_non_csv`: Confirms file type validation.
- `test_csv_upload_valid_processing`: Tests batch processing and batch record tracking.
- `test_topic_discovery_analytics`: Tests K-Means clustering and low-sample warnings.
- `test_business_insights_generation`: Tests rule-based recommendation logic.
- `test_model_evaluation_page`: Verifies test benchmark rendering.

---

## 7. Technical Interview Preparation & Defense

### Q1: Why build this in Django instead of Streamlit?
**Answer:** Streamlit is suited for rapid prototypes and data science scripts, but it re-runs entire Python scripts top-to-bottom on every user interaction, making it unsuitable for multi-user transactional software. Django provides a robust Model-View-Template (MVT) architecture, production-ready ORM, database connection pooling, user authentication, CSRF security, asynchronous background task compatibility, and scalable REST API extensibility.

### Q2: How does the CSV upload handle large files without crashing?
**Answer:**
1. File size is capped at 10MB in Django settings and forms.
2. The file is streamed and parsed safely in memory via pandas.
3. Sentiment predictions are batched across vector representations in memory rather than single iterations.
4. Database writes use `Review.objects.bulk_create(..., batch_size=500)` to execute single multi-row SQL `INSERT` statements, reducing database roundtrips from thousands to a few queries.
5. All upload lifecycle states are tracked inside an `UploadBatch` model with error handling to isolate bad uploads.

### Q3: How do TF-IDF and Logistic Regression work in this pipeline?
**Answer:**
- **TF-IDF (Term Frequency - Inverse Document Frequency)** transforms raw text into a high-dimensional sparse numerical matrix ($25,000$ dimensions). TF rewards words frequent in a specific review, while IDF penalizes ubiquitous words common across all reviews (e.g. "product", "the"). Sublinear TF scaling ($1 + \log(\text{TF})$) prevents long reviews from skewing weights.
- **Logistic Regression** applies a linear decision boundary on the sparse TF-IDF vectors passed through the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to estimate class probabilities $P(y=1|X)$. It converges fast, trains in seconds on CPU, and resists overfitting via $L_2$ regularization.

### Q4: How does K-Means topic clustering operate?
**Answer:** K-Means partitions reviews into $k$ clusters in the TF-IDF feature space by minimizing the within-cluster sum of squares (inertia):
$$\arg\min_S \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$$
Each cluster's centroid vector $\mu_i$ represents the average TF-IDF weighting of words in that cluster. By sorting the centroid dimensions in descending order, we extract the highest-weighted terms as characteristic keywords describing that topic.

### Q5: What are the model limitations and future improvements?
**Answer:**
- **Context & Sarcasm:** Bag-of-words and n-grams cannot easily capture complex negated syntax or subtle sarcasm (e.g., "Just what I needed, another broken gadget"). We mitigate this by flagging low-confidence predictions ($<65\%$).
- **Neutral Class:** The binary dataset classifies polarity; adding a dedicated 3-class neutral label would capture strictly informational feedback.
- **Future Improvements:** Fine-tuning a lightweight local transformer (e.g. DistilRoBERTa via ONNX runtime) and implementing Celery/Redis for asynchronous CSV processing.

---

## 8. Uploading to GitHub

```powershell
cd D:\voice
git init
git add .
git commit -m "feat: complete Voice of Customer AI platform with Django 5.2 and Scikit-learn"
git branch -M main
git remote add origin https://github.com/<your-username>/voice-of-customer-ai.git
git push -u origin main
```
