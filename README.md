# Gastrointestinal Symptom Analysis Models

A comprehensive machine learning analysis of gastrointestinal symptoms using various neural network and ensemble approaches.

## Models Implemented

1. **Graph Neural Network (GNN)**
   - Uses symptom co-occurrence patterns
   - Implements attention mechanisms
   - Captures complex symptom interactions
   - Features batch normalization and dropout

2. **Feed-Forward Neural Network (FFNN)**
   - Multi-layer architecture
   - Feature attention mechanism
   - Abundance and interaction processing
   - Dropout regularization

3. **Random Forest (RF)**
   - Ensemble of 100 trees
   - Feature importance analysis
   - Optimized hyperparameters
   - Parallel processing enabled

4. **AdaBoost (AB)**
   - Decision tree base estimators
   - Adaptive boosting strategy
   - Depth-3 weak learners
   - Learning rate optimization

## Features Analyzed

### Symptom Frequency (Days/Week)
- Acidity
- Bloating
- Flatulence
- Constipation
- Burping

### Impact on Daily Life (Scale 1-10)
- Bloating
- Acidity/Burning
- Constipation
- Loose Motion/Diarrhea
- Flatulence
- Burping

## Requirements

```python
# Core Dependencies
numpy>=1.19.2
pandas>=1.2.4
scikit-learn>=0.24.2

# Deep Learning
torch>=1.8.1
torch_geometric>=2.0.0

# Visualization
matplotlib>=3.3.4

# Utilities
tqdm>=4.61.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gi-symptom-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Contact repository owner to obtain the `dataset_new.csv`
2. Place dataset in project root directory
3. Run individual models:

```bash
# Graph Neural Network
python graph_neural_network.py

# Random Forest Analysis
python random_forest_analysis.py

# AdaBoost Analysis
python adaboost_analysis.py
```

## Model Details

### Data Preprocessing
- Standard scaling
- PCA dimensionality reduction
- Label encoding
- Missing value handling

### Training Features
- Early stopping
- Learning rate scheduling
- K-means clustering for balanced sampling
- Cross-validation

### Evaluation Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- ROC Curves (for GNN)

## Results Storage

Each model saves results in a structured format:

```
results/
├── model_name_timestamp/
│   ├── model.pt (or .pkl)
│   ├── metrics.json
│   ├── metadata.npz
│   └── visualizations/
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For dataset access and additional information, please contact the repository owner.

## Acknowledgments

- Dataset provided by [Institution Name]
- Based on research by [Research Group/Paper]

**Note**: This is a research project dealing with medical data. The dataset contains sensitive information and is not publicly available.
