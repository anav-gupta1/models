import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import random
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import torch.jit
import os
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SymptomGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(SymptomGNN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        
        if x.dim() == 3:
            x = x.squeeze(1)
        
        num_nodes = x.size(0)
        edge_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, edge_mask]
        edge_weight = edge_weight[edge_mask] if edge_weight is not None else None
        
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        
        
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)
        
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def construct_cooccurrence_graph(features, threshold=0.3):
    
    n_features = features.shape[1]
    
    
    cooccurrence_matrix = np.zeros((n_features, n_features))
    
    
    support = np.sum(features > 0, axis=0) / len(features)
    
    
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                cooccurrence = np.sum((features[:, i] > 0) & 
                                    (features[:, j] > 0)) / len(features)
                
                
                if support[i] * support[j] > 0:
                    lift = cooccurrence / (support[i] * support[j])
                else:
                    lift = 0
                    
                
                jaccard = cooccurrence / (support[i] + support[j] - cooccurrence + 1e-10)
                
                
                combined_score = 0.7 * lift + 0.3 * jaccard
                
                cooccurrence_matrix[i, j] = combined_score
    
    
    edges = np.where(cooccurrence_matrix > threshold)
    edge_weights = cooccurrence_matrix[edges]
    
    
    self_loops = np.array([[i, i] for i in range(n_features)])
    self_loop_weights = np.ones(n_features)
    
    
    if len(edges[0]) > 0:
        all_edges = np.concatenate([np.vstack(edges).T, self_loops], axis=0)
        all_weights = np.concatenate([edge_weights, self_loop_weights])
    else:
        all_edges = self_loops
        all_weights = self_loop_weights
    
    
    max_index = n_features - 1
    valid_edges_mask = (all_edges[:, 0] <= max_index) & (all_edges[:, 1] <= max_index)
    all_edges = all_edges[valid_edges_mask]
    all_weights = all_weights[valid_edges_mask]
    
    
    edge_index = torch.tensor(all_edges.T, dtype=torch.long)
    edge_weight = torch.tensor(all_weights, dtype=torch.float)
    
    return edge_index, edge_weight

def cluster_based_oversampling(features, labels, cluster_labels, minority_class):
    
    
    features = np.array(features)
    labels = np.array(labels)
    cluster_labels = np.array(cluster_labels)
    
    
    class_counts = np.bincount(labels)
    majority_count = np.max(class_counts)
    minority_count = class_counts[minority_class]
    
    
    samples_to_generate = majority_count - minority_count
    
    
    minority_mask = labels == minority_class
    minority_samples = features[minority_mask]
    minority_clusters = cluster_labels[minority_mask]
    
    
    features_resampled = list(features)
    labels_resampled = list(labels)
    
    
    unique_clusters = np.unique(minority_clusters)
    samples_per_cluster = samples_to_generate // len(unique_clusters)
    remainder = samples_to_generate % len(unique_clusters)
    
    for cluster in unique_clusters:
        cluster_samples = minority_samples[minority_clusters == cluster]
        num_samples_in_cluster = len(cluster_samples)
        
        if num_samples_in_cluster < 2:
            continue
        
        
        num_samples_to_generate = samples_per_cluster
        if remainder > 0:
            num_samples_to_generate += 1
            remainder -= 1
        
        
        for _ in range(num_samples_to_generate):
            idx1, idx2 = random.sample(range(num_samples_in_cluster), 2)
            synthetic_sample = (cluster_samples[idx1] + cluster_samples[idx2]) / 2
            features_resampled.append(synthetic_sample)
            labels_resampled.append(minority_class)
    
    return np.array(features_resampled), np.array(labels_resampled)

def prepare_graph_data(df, symptom_column, n_components=128, threshold=0.3):
    
    
    X = df.drop(columns=[symptom_column])
    y = df[symptom_column]
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    
    print("\nPerforming clustering for oversampling...")
    kmeans = KMeans(n_clusters=min(10, len(X_pca)), random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    
    minority_class_index = np.argmin(np.bincount(y_encoded))  
    X_resampled, y_resampled = cluster_based_oversampling(
        X_pca, y_encoded, cluster_labels, minority_class=minority_class_index
    )
    
    
    edge_index, edge_weight = construct_cooccurrence_graph(X_resampled, threshold)
    
    
    graph_data_list = []
    for i in range(len(X_resampled)):
        
        x = torch.tensor(X_resampled[i], dtype=torch.float)
        if x.dim() == 1:
            x = x.unsqueeze(0)  
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([y_resampled[i]], dtype=torch.long)
        )
        graph_data_list.append(data)
    
    return graph_data_list, label_encoder, scaler, pca

def plot_roc_curves(y_true, y_pred_proba, label_encoder, save_path):
    
    plt.figure(figsize=(10, 8))
    
    
    y_test_bin = label_binarize(y_true, classes=range(len(label_encoder.classes_)))
    
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i],
                label=f'ROC curve class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    
    return roc_auc

def analyze_symptom_severity_gnn(data_path, symptom_name, hidden_dim=128, n_components=128):
    
    print(f"\nStarting GNN analysis for {symptom_name.upper()} severity...")
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.fillna(0)
    
    symptom_column = "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Burping]"
    
    
    graph_data_list, label_encoder, scaler, pca = prepare_graph_data(
        df, symptom_column, n_components
    )
    
    
    y_all = [data.y.item() for data in graph_data_list]
    print("\nClass distribution after oversampling:")
    for class_label in np.unique(y_all):
        count = sum(1 for y in y_all if y == class_label)
        print(f"Class {label_encoder.inverse_transform([class_label])[0]}: {count}")
    
    
    train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)
    
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)
    
    
    num_features = graph_data_list[0].x.size(1)
    num_classes = len(label_encoder.classes_)
    model = SymptomGNN(num_features, hidden_dim, num_classes).to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    
    best_accuracy = 0
    patience = 15
    patience_counter = 0
    start_time = time.time()
    
    print("\nStarting training...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        train_pred = []
        train_true = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            try:
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                train_pred.extend(pred.cpu().numpy())
                train_true.extend(batch.y.cpu().numpy())
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                continue
        
        
        model.eval()
        test_pred = []
        test_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                try:
                    out = model(batch)
                    pred = out.argmax(dim=1)
                    test_pred.extend(pred.cpu().numpy())
                    test_true.extend(batch.y.cpu().numpy())
                except RuntimeError as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        
        if train_pred and test_pred:  
            train_acc = accuracy_score(train_true, train_pred)
            test_acc = accuracy_score(test_true, test_pred)
            
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch + 1}/100:")
                print(f"Loss: {total_loss / len(train_loader):.4f}")
                print(f"Train Accuracy: {train_acc:.4f}")
                print(f"Test Accuracy: {test_acc:.4f}")
            
            scheduler.step(test_acc)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("\nEarly stopping triggered!")
                    break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    final_pred = []
    final_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            try:
                out = model(batch)
                pred = out.argmax(dim=1)
                final_pred.extend(pred.cpu().numpy())
                final_true.extend(batch.y.cpu().numpy())
            except RuntimeError as e:
                print(f"Error in final evaluation: {e}")
                continue
    
    final_pred_proba = []
    final_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            try:
                out = model(batch)
                pred_proba = torch.softmax(out, dim=1)
                final_pred_proba.extend(pred_proba.cpu().numpy())
                final_true.extend(batch.y.cpu().numpy())
            except RuntimeError as e:
                print(f"Error in final evaluation: {e}")
                continue
    
    final_pred_proba = np.array(final_pred_proba)
    final_true = np.array(final_true)
    
    
    save_dir = os.path.join("results", f"symptom_gnn_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    
    
    roc_curve_path = os.path.join(save_dir, 'roc_curves.png')
    roc_auc_scores = plot_roc_curves(final_true, final_pred_proba, label_encoder, roc_curve_path)
    
    
    model.eval()
    traced_model = torch.jit.script(model)
    traced_model.save(os.path.join(save_dir, 'model.pt'))
    
    
    metadata = {
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'pca_components': pca.components_.tolist(),
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'roc_auc': {str(k): v for k, v in roc_auc_scores.items()},
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
    }
    
    
    np.savez_compressed(
        os.path.join(save_dir, 'metadata.npz'),
        **{k: np.array(v) if isinstance(v, list) else v 
           for k, v in metadata.items()}
    )
    
    print(f"\nResults saved in: {save_dir}")
    print("Files saved:")
    print(f" - Model: model.pt (TorchScript format)")
    print(f" - Metadata: metadata.npz (compressed)")
    print(f" - ROC Curves: roc_curves.png")
    
    return model, metrics, label_encoder, scaler, pca

def load_model_and_metadata(save_dir):
    
    
    model = torch.jit.load(os.path.join(save_dir, 'model.pt'))
    
    
    metadata = np.load(os.path.join(save_dir, 'metadata.npz'), allow_pickle=True)
    
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = metadata['label_encoder_classes']
    
    pca = PCA(n_components=len(metadata['pca_explained_variance']))
    pca.components_ = metadata['pca_components']
    pca.explained_variance_ratio_ = metadata['pca_explained_variance']
    
    scaler = StandardScaler()
    scaler.mean_ = metadata['scaler_mean']
    scaler.scale_ = metadata['scaler_scale']
    
    metrics = metadata['metrics'].item()
    
    return model, metrics, label_encoder, scaler, pca

if __name__ == "__main__":
    
    data_path = 'dataset_new.csv' 
    
    
    model, metrics, label_encoder, scaler, pca = analyze_symptom_severity_gnn(
        data_path, 'symptom_severity'
    )
    
    
    print("\nDetailed Results:")
    print("=" * 50)
    print(f"Model Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    print("-" * 50)
    print(metrics['confusion_matrix'])
    
    
    print("\nClass Mapping:")
    print("-" * 50)
    for i, label in enumerate(label_encoder.classes_):
        print(f"Class {i}: {label}")
    
    
    try:
        feature_names = list(pd.read_csv(data_path).drop(columns=["How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Burping]", "SAMPLE BARCODE"]).columns)
    except:
        feature_names = [f"Feature_{i}" for i in range(model.conv1.in_channels)]
    
    
    print("\nGenerating graph visualization...")
    
    sample_data = next(iter(DataLoader(graph_data_list, batch_size=1)))
    visualize_graph_structure(
        sample_data.edge_index,
        sample_data.edge_weight,
        feature_names=feature_names
    )
    
    
    print("\nAnalyzing important symptom connections...")
def analyze_important_connections(edge_index, edge_weight, feature_names, top_k=15):
    # Function implementation here
    
    
    save_dir = "results/symptom_gnn_" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    
    
    print(f"\nSaving results to {save_dir}...")
    
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'label_encoder_classes': label_encoder.classes_,
        'pca_components': pca.components_,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, os.path.join(save_dir, 'model_checkpoint.pt'))
    
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }, f, indent=4)
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pca_explained_variance.png'))
    plt.close()
    
    print("\nAnalysis complete! Results have been saved.")
    print(f"Model and results saved in: {save_dir}")
    
    
    print("\nModel Architecture:")
    print("-" * 50)
    print(model)