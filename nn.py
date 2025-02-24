import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def cluster_based_oversampling(X, y, cluster_labels, minority_class=0):
    X_resampled = list(X)
    y_resampled = list(y)
    minority_samples = X[y == minority_class]
    minority_clusters = cluster_labels[y == minority_class]

    for cluster in np.unique(minority_clusters):
        cluster_samples = minority_samples[minority_clusters == cluster]
        num_samples_in_cluster = len(cluster_samples)

        if num_samples_in_cluster < 2:
            X_resampled.extend(cluster_samples)
            y_resampled.extend([minority_class] * num_samples_in_cluster)
            continue

        num_samples_to_generate = num_samples_in_cluster
        for _ in range(num_samples_to_generate):
            sample_1, sample_2 = random.sample(list(cluster_samples), 2)
            synthetic_sample = (sample_1 + sample_2) / 2
            X_resampled.append(synthetic_sample)
            y_resampled.append(minority_class)

    return np.array(X_resampled), np.array(y_resampled)

class MatrixBasedAcidityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.abundance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.interaction_layer = nn.Bilinear(input_dim, input_dim, hidden_dim)
        
        self.interaction_network = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )

        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        feature_weights = torch.sigmoid(self.feature_attention(x))
        weighted_input = x * feature_weights
        
        abundance_features = self.abundance_network(weighted_input)
        
        interaction_features = self.interaction_layer(weighted_input, weighted_input)
        interaction_features = self.interaction_network(interaction_features)
        
        combined = torch.cat([abundance_features, interaction_features], dim=1)
        output = self.final_layers(combined)
        return output

def analyze_symptom_severity_matrix(data_path, symptom_name, n_components=1024, patience=10):
    print(f"\nStarting analysis for {symptom_name.upper()}...")
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
    n_classes = len(label_encoder.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    X_resampled, y_resampled = cluster_based_oversampling(X_pca, y_encoded, clusters)

    train_size = int(0.8 * len(X_resampled))
    indices = np.random.permutation(len(X_resampled))
    
    X_train = X_resampled[indices[:train_size]]
    y_train = y_resampled[indices[:train_size]]
    X_val = X_resampled[indices[train_size:]]
    y_val = y_resampled[indices[train_size:]]
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    model = MatrixBasedAcidityNet(input_dim=n_components, hidden_dim=128, output_dim=n_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    model.eval()
    with torch.no_grad():
        final_preds = []
        final_labels = []
        for features, labels in val_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            final_preds.extend(predicted.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
    
    print("\nFinal Results:")
    print(f"Accuracy: {accuracy_score(final_labels, final_preds):.4f}")
    print(f"Precision: {precision_score(final_labels, final_preds, average='weighted'):.4f}")
    print(f"Recall: {recall_score(final_labels, final_preds, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(final_labels, final_preds, average='weighted'):.4f}")

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
        
        analyze_symptom_severity_matrix("dataset_new.csv", symptom_column)