#!/usr/bin/env python
"""
prepare_data.py
Download and prepare a real Amazon review dataset for training the Voice of Customer AI model.
Uses Hugging Face Datasets with streaming or fallback to a sample to ensure reliability and safety.
"""

import os
import sys
import pandas as pd

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'amazon_reviews.csv')
TARGET_SIZE = 12000  # 6,000 positive and 6,000 negative reviews

def prepare_amazon_reviews():
    print("=" * 60)
    print("VOICE OF CUSTOMER AI - DATA PREPARATION")
    print("=" * 60)
    print(f"Target sample size: {TARGET_SIZE} reviews (~50% positive, ~50% negative)")
    print(f"Output path: {OUTPUT_FILE}")

    reviews = []
    labels = []

    pos_target = TARGET_SIZE // 2
    neg_target = TARGET_SIZE // 2
    pos_count = 0
    neg_count = 0

    try:
        from datasets import load_dataset
        print("\nAttempting to stream 'fancyzhx/amazon_polarity' or 'amazon_polarity' from Hugging Face...")
        
        dataset = None
        for ds_name in ["fancyzhx/amazon_polarity", "amazon_polarity"]:
            try:
                print(f"Connecting to Hugging Face dataset: '{ds_name}' (split='train', streaming=True)...")
                dataset = load_dataset(ds_name, split='train', streaming=True)
                break
            except Exception as e:
                print(f"Could not load '{ds_name}': {e}")

        if dataset is not None:
            print("Streaming reviews from Hugging Face...")
            for item in dataset:
                # Amazon Polarity label: 0 -> Negative, 1 -> Positive
                # Fields usually: 'title', 'content' or 'text'
                title = item.get('title', '') or ''
                content = item.get('content', '') or item.get('text', '') or ''
                full_text = f"{title}. {content}".strip() if title else content.strip()
                label_val = item.get('label')

                if not full_text or len(full_text) < 15:
                    continue

                if label_val == 1 and pos_count < pos_target:
                    reviews.append(full_text)
                    labels.append('Positive')
                    pos_count += 1
                elif label_val == 0 and neg_count < neg_target:
                    reviews.append(full_text)
                    labels.append('Negative')
                    neg_count += 1

                if pos_count >= pos_target and neg_count >= neg_target:
                    break
                    
                if (pos_count + neg_count) % 2000 == 0 and (pos_count + neg_count) > 0:
                    print(f"Fetched {pos_count + neg_count}/{TARGET_SIZE} reviews...")

    except Exception as exc:
        print(f"Hugging Face streaming encountered an error or network limitation: {exc}")

    # Fallback to authentic public Amazon sample dataset if HF streaming had connection errors
    if len(reviews) < 1000:
        print("\nHugging Face streaming incomplete or offline. Loading verified authentic Amazon reviews via direct sample...")
        # Direct URL or verified public mirror fallback
        import urllib.request
        import json
        import random

        # We can fetch from raw public domain / curated reviews URL or generate a rich verified seed
        fallback_url = "https://raw.githubusercontent.com/Ankit152/Amazon-Reviews-Sentiment-Analysis/master/amazon_reviews.csv"
        try:
            print(f"Downloading from verified public mirror: {fallback_url}")
            df_mirror = pd.read_csv(fallback_url, nrows=20000)
            # Find review and sentiment columns
            text_col = None
            for c in ['reviewText', 'verified_reviews', 'review', 'Text', 'body']:
                if c in df_mirror.columns:
                    text_col = c
                    break
            sentiment_col = None
            for c in ['feedback', 'sentiment', 'overall', 'Score', 'rating']:
                if c in df_mirror.columns:
                    sentiment_col = c
                    break

            if text_col and sentiment_col:
                for _, row in df_mirror.iterrows():
                    txt = str(row[text_col]).strip()
                    val = row[sentiment_col]
                    if not txt or len(txt) < 15 or txt.lower() == 'nan':
                        continue
                    # map rating/feedback
                    if val in [1, 'positive', 'Positive', 4, 5, 4.0, 5.0]:
                        if pos_count < pos_target:
                            reviews.append(txt)
                            labels.append('Positive')
                            pos_count += 1
                    elif val in [0, 'negative', 'Negative', 1.0, 2.0, 1, 2]:
                        if neg_count < neg_target:
                            reviews.append(txt)
                            labels.append('Negative')
                            neg_count += 1
                    if pos_count >= pos_target and neg_count >= neg_target:
                        break
        except Exception as e2:
            print(f"Could not load from mirror: {e2}")

    # Ensure output
    if len(reviews) == 0:
        print("ERROR: Could not fetch data. Please check internet connection.")
        sys.exit(1)

    df = pd.DataFrame({
        'review': reviews,
        'sentiment': labels
    })

    # Data hygiene: remove duplicates & nulls
    initial_len = len(df)
    df.dropna(subset=['review', 'sentiment'], inplace=True)
    df.drop_duplicates(subset=['review'], inplace=True)
    print(f"\nCleaned dataset: {len(df)} reviews (removed {initial_len - len(df)} duplicates/nulls).")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nSaved clean dataset successfully to: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == '__main__':
    prepare_amazon_reviews()
