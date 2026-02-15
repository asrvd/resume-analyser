#!/usr/bin/env python3
"""
Simple Resume Classifier - No SetFit Dependency Hell
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json
import os

def train_model():
    print("="*70)
    print("TRAINING RESUME CLASSIFIER")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    test_df = pd.read_csv('data/test.csv')
    
    with open('data/label_map.json', 'r') as f:
        label_map = json.load(f)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load embedding model
    print("\nLoading embedding model (first run downloads ~90MB)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    train_emb = model.encode(train_df['text'].tolist(), show_progress_bar=True)
    val_emb = model.encode(val_df['text'].tolist(), show_progress_bar=True)
    test_emb = model.encode(test_df['text'].tolist(), show_progress_bar=True)
    
    # Train
    print("\nTraining classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(train_emb, train_df['label'])
    
    # Evaluate
    val_pred = clf.predict(val_emb)
    val_acc = accuracy_score(val_df['label'], val_pred)
    print(f"Validation Accuracy: {val_acc:.2%}")
    
    test_pred = clf.predict(test_emb)
    test_acc = accuracy_score(test_df['label'], test_pred)
    
    print(f"\n{'='*70}")
    print(f"TEST ACCURACY: {test_acc:.2%}")
    print(f"{'='*70}\n")
    
    print(classification_report(test_df['label'], test_pred, 
                                target_names=list(label_map.keys())))
    
    # Save
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/sentence_transformer')
    
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    with open('models/label_map.json', 'w') as f:
        json.dump(label_map, f)
    
    print("✓ Model saved to models/")
    print(f"✓ Final accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    train_model()