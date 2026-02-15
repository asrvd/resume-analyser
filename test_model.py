#!/usr/bin/env python3
"""
Test the trained model on sample resumes
"""

from sentence_transformers import SentenceTransformer
import pickle
import json
import pandas as pd


def load_model():
    """Load trained model"""
    print("Loading model...")

    embedder = SentenceTransformer("models/sentence_transformer")

    with open("models/classifier.pkl", "rb") as f:
        classifier = pickle.load(f)

    with open("models/label_map.json", "r") as f:
        label_map = json.load(f)

    id_to_label = {v: k for k, v in label_map.items()}

    print("✓ Model loaded successfully\n")
    return embedder, classifier, id_to_label


def predict_resume(text, embedder, classifier, id_to_label):
    """Predict category for a resume"""

    # Generate embedding
    embedding = embedder.encode([text])

    # Predict
    pred_id = classifier.predict(embedding)[0]
    probs = classifier.predict_proba(embedding)[0]

    # Get category name
    category = id_to_label[pred_id]
    confidence = probs[pred_id]

    # All probabilities
    all_probs = {id_to_label[i]: prob for i, prob in enumerate(probs)}

    return category, confidence, all_probs


def test_on_samples():
    """Test on sample resumes from test set"""

    # Load model
    embedder, classifier, id_to_label = load_model()

    # Load test data
    test_df = pd.read_csv("data/test.csv")

    print("=" * 70)
    print("TESTING MODEL ON SAMPLE RESUMES")
    print("=" * 70)

    # Test on first 5 resumes
    for i in range(min(5, len(test_df))):
        sample = test_df.iloc[i]

        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1}")
        print(f"{'='*70}")

        # Show resume preview
        print(f"\nResume Preview:")
        print(sample["text"][:300] + "...\n")

        # True category
        print(f"True Category: {sample['category']}")

        # Predict
        category, confidence, all_probs = predict_resume(
            sample["text"], embedder, classifier, id_to_label
        )

        # Show prediction
        print(f"\nPredicted Category: {category}")
        print(f"Confidence: {confidence:.1%}")

        print(f"\nAll Probabilities:")
        for cat, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 50)
            print(f"  {cat:20} {prob:.1%} {bar}")

        # Result
        if category == sample["category"]:
            print("\n✓ CORRECT")
        else:
            print("\n✗ INCORRECT")

    print(f"\n{'='*70}")
    print("Testing complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_on_samples()
