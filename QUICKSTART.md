# Quick Start Guide

Get started with DistAwareAug in 5 minutes!

## Installation

```bash
pip install -e .
```

## Basic Example

```python
from distawareaug import DistAwareAugmentor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.9, 0.1],  # 90% majority, 10% minority
    random_state=42
)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Scale features (IMPORTANT for diversity control!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create augmentor
augmentor = DistAwareAugmentor(
    sampling_strategy='auto',      # Balance all classes
    diversity_threshold=0.1,       # Minimum distance between samples
    distribution_method='kde',     # 'kde' or 'gaussian'
    random_state=42
)

# 5. Oversample minority class
X_resampled, y_resampled = augmentor.fit_resample(X_train_scaled, y_train)

# 6. Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# 7. Evaluate
score = clf.score(X_test_scaled, y_test)
print(f"Test Accuracy: {score:.4f}")
print(f"Original training samples: {len(X_train)}")
print(f"Augmented training samples: {len(X_resampled)}")
```

## Key Parameters

### `sampling_strategy`
- `'auto'`: Balance all classes to majority class size
- `dict`: Custom, e.g., `{0: 0, 1: 500}` generates 500 samples for class 1

### `diversity_threshold`
- Higher (0.2-0.5): More diverse samples, fewer generated
- Lower (0.05-0.1): More samples generated, less diversity
- **Important**: Scale your features first!

### `distribution_method`
- `'kde'`: Kernel Density Estimation (more accurate, slower)
- `'gaussian'`: Multivariate Gaussian (faster, ~10x)

### `distance_metric`
- `'euclidean'`: Standard Euclidean distance (default)
- `'manhattan'`: L1 distance
- `'cosine'`: Cosine similarity

## Common Workflows

### Workflow 1: Quick Balancing

```python
from distawareaug import DistAwareAugmentor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

augmentor = DistAwareAugmentor(random_state=42)
X_resampled, y_resampled = augmentor.fit_resample(X_scaled, y)
```

### Workflow 2: Custom Sample Count

```python
# Generate exactly 500 samples for class 1, leave class 0 as-is
augmentor = DistAwareAugmentor(
    sampling_strategy={0: 0, 1: 500},
    random_state=42
)
X_resampled, y_resampled = augmentor.fit_resample(X_scaled, y)
```

### Workflow 3: Fast Mode (Gaussian)

```python
# For large datasets, use Gaussian distribution
augmentor = DistAwareAugmentor(
    distribution_method='gaussian',  # 10x faster than KDE
    random_state=42
)
X_resampled, y_resampled = augmentor.fit_resample(X_scaled, y)
```

### Workflow 4: Analyze Diversity

```python
from distawareaug import DistAwareAugmentor, DistanceMetrics

augmentor = DistAwareAugmentor(random_state=42)
X_resampled, y_resampled = augmentor.fit_resample(X_scaled, y)

# Get synthetic samples (those added after original)
n_original = len(X)
synthetic_samples = X_resampled[n_original:]

# Calculate diversity score
dm = DistanceMetrics(metric='euclidean')
diversity = dm.diversity_score(synthetic_samples)
print(f"Diversity Score: {diversity:.4f}")
```

## Troubleshooting

### Problem: Too few samples generated

**Cause**: `diversity_threshold` is too high for your data scale

**Solution**: Scale your features!
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problem: Slow performance

**Cause**: KDE is computationally expensive

**Solution**: Use Gaussian method
```python
augmentor = DistAwareAugmentor(distribution_method='gaussian')
```

### Problem: Low diversity

**Cause**: `diversity_threshold` is too low

**Solution**: Increase threshold
```python
augmentor = DistAwareAugmentor(diversity_threshold=0.2)
```

## Next Steps

1. Explore the `examples/` directory for detailed notebooks
2. Read the [full documentation](README.md)
3. Compare with SMOTE: `examples/compare_smote.ipynb`
4. Join discussions on GitHub

## Quick Reference

```python
# Import
from distawareaug import DistAwareAugmentor

# Create
augmentor = DistAwareAugmentor(
    sampling_strategy='auto',
    diversity_threshold=0.1,
    distribution_method='kde',
    distance_metric='euclidean',
    random_state=42
)

# Use
X_resampled, y_resampled = augmentor.fit_resample(X, y)
```

That's it! You're ready to use DistAwareAug. 🎉
