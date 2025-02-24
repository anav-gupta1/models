import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def analyze_symptom_severity_rf(data_path, symptom_name, n_components=1024):
    print(f"\nStarting Random Forest analysis for {symptom_name.upper()}...")
    set_seed(42)


    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.fillna(0)

    severity_column = symptom_name
    X = df.drop(columns=[severity_column])
    y = df[severity_column]

    print("\nInitial class distribution:")
    print(y.value_counts().sort_index())
    

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)


    X_train, X_val, y_train, y_val = train_test_split(
        X_pca, y_encoded, test_size=0.2, random_state=42
    )


    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    

    y_pred = rf_model.predict(X_val)
    
    print("\nFinal Results:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred, average='weighted'):.4f}")

if __name__ == "__main__":
    symptom_columns = [
        "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Acidity]",
        "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Bloating]",
        "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Flatulence/Gas/Fart]",
        "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Constipation]",
        "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Burping]",
        "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Bloating]",
        "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Acidity/Burning]",
        "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Constipation]",
        "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Loose Motion/Diarrhea]",
        "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Flatulence/Gas/Fart]",
        "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Burping]"
    ]

    for symptom_column in symptom_columns:
        print("\n" + "="*80)
        print(f"Analyzing: {symptom_column}")
        print("="*80)
        
        analyze_symptom_severity_rf("dataset_new.csv", symptom_column) 