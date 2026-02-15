import pandas as pd
from sklearn.model_selection import train_test_split
import json


def preprocess_dataset():
    """Prepare data for model training"""

    print("Loading dataset...")
    df = pd.read_csv("output/csv/synthetic_resumes_500.csv")

    # Create label mapping
    label_map = {
        "Private Job": 0,
        "Higher Studies": 1,
        "Research Field": 2,
        "Skill Improvement": 3,
    }

    df["label"] = df["category"].map(label_map)

    # Split data: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Save splits
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    # Save label mapping
    with open("data/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("\n✓ Data preprocessing complete!")
    print("✓ Files saved in data/ directory")

    return train_df, val_df, test_df


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)

    train_df, val_df, test_df = preprocess_dataset()
