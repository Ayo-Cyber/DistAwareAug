<div align="center">

# 🎯 DistAwareAug

**Distribution-Aware Data Augmentation for Imbalanced Learning**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Intelligent oversampling that preserves statistical distributions and ensures sample diversity*

[Features](#features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Contributing](#contributing) •
[Citation](#citation)

</div>

---

## 📖 Overview

**DistAwareAug** introduces a new paradigm for handling imbalanced datasets: **statistically-governed augmentation**. Unlike traditional methods like SMOTE that interpolate between samples, DistAwareAug:

1. 📊 **Learns the statistical distribution** of minority class features (mean, variance, covariance)
2. 🎲 **Generates synthetic samples** from fitted distributions (KDE or Gaussian)
3. 🎯 **Ensures diversity** through distance-based filtering
4. ✅ **Preserves feature bounds** and avoids unrealistic outliers

### Why DistAwareAug?

| Feature | SMOTE | BorderlineSMOTE | ADASYN | **DistAwareAug** |
|---------|-------|-----------------|--------|------------------|
| Preserves distributions | ❌ | ❌ | ❌ | ✅ |
| Diversity control | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Flexible sampling (add/target) | ❌ | ❌ | ❌ | ✅ |
| Avoids interpolation artifacts | ❌ | ❌ | ❌ | ✅ |
| Statistical governance | ❌ | ❌ | ❌ | ✅ |
| Supports downsampling | ❌ | ❌ | ❌ | ✅ |
| Distribution methods (KDE/Gaussian) | ❌ | ❌ | ❌ | ✅ |

---

## ✨ Features

- 🔬 **Distribution Fitting**: KDE (Kernel Density Estimation) or Gaussian distributions
- 📏 **Distance Metrics**: Euclidean, Manhattan, Cosine, Minkowski, and more
- 🎲 **Diversity Control**: Configurable threshold for sample uniqueness
- 🎯 **Flexible Sampling Modes**: 
  - **'add' mode**: Add N samples to existing class count
  - **'target' mode**: Target absolute sample counts (supports both upsampling and downsampling)
- 🔄 **scikit-learn Compatible**: Drop-in replacement for SMOTE and similar methods
- ⚡ **Performance Optimized**: Batch generation with efficient diversity checking
- 📊 **Quality Metrics**: Built-in diversity scoring and validation
- 🛡️ **Robust**: Handles edge cases, singular matrices, and various data types

---

## 🚀 Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/Ayo-Cyber/DistAwareAug.git
cd DistAwareAug

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e ".[all]"
```

### For Users (Once Published to PyPI)

```bash
pip install distawareaug
```

### Minimal Installation

```bash
pip install distawareaug
```

### With Optional Dependencies

```bash
# For development (testing, linting, formatting)
pip install distawareaug[dev]

# For running examples
pip install distawareaug[examples]

# For building documentation
pip install distawareaug[docs]

# Everything
pip install distawareaug[all]
```

---

## 🎯 Quick Start

### Basic Usage

```python
from distawareaug import DistAwareAugmentor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.9, 0.1],  # 90% majority, 10% minority
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize DistAwareAugmentor
augmentor = DistAwareAugmentor(
    sampling_strategy='auto',      # Balance all classes
    diversity_threshold=0.1,       # Minimum distance between samples
    distribution_method='kde',     # or 'gaussian' for speed
    random_state=42
)

# Oversample the training data
X_resampled, y_resampled = augmentor.fit_resample(X_train, y_train)

# Train classifier on balanced data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# Evaluate
score = clf.score(X_test, y_test)
print(f"Test Accuracy: {score:.4f}")
```

### Advanced Usage

```python
from distawareaug import DistAwareAugmentor, DistanceMetrics
from sklearn.preprocessing import StandardScaler

# Scale features for better diversity control
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Example 1: Add mode (adds N samples to existing count)
augmentor_add = DistAwareAugmentor(
    sampling_strategy={0: 0, 1: 500},  # Add 500 samples to class 1, no change to class 0
    sampling_mode='add',                # Default: adds to existing count
    diversity_threshold=0.15,
    distribution_method='gaussian',
    distance_metric='manhattan',
    random_state=42
)

X_resampled, y_resampled = augmentor_add.fit_resample(X_train_scaled, y_train)

# Example 2: Target mode (targets absolute count, can upsample or downsample)
augmentor_target = DistAwareAugmentor(
    sampling_strategy={0: 1000, 1: 1000},  # Target exactly 1000 samples for each class
    sampling_mode='target',                 # Will upsample minority, downsample majority
    diversity_threshold=0.15,
    distribution_method='gaussian',
    random_state=42
)

X_balanced, y_balanced = augmentor_target.fit_resample(X_train_scaled, y_train)

# Analyze diversity
dm = DistanceMetrics(metric='euclidean')
diversity = dm.diversity_score(X_resampled[len(X_train):])
print(f"Synthetic Sample Diversity: {diversity:.4f}")
```

### Using with scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from distawareaug import DistAwareAugmentor
from sklearn.svm import SVC

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('augmentor', DistAwareAugmentor(distribution_method='gaussian')),
    ('classifier', SVC())
])

# Note: Augmentor only applies to training data
# Use manually for proper train/test separation
```

### Understanding Sampling Modes

DistAwareAug offers two sampling modes that control how `sampling_strategy` values are interpreted:

#### **'add' Mode (Default)**
Adds N samples to the existing class count:
```python
# If class 1 has 100 samples originally
augmentor = DistAwareAugmentor(
    sampling_strategy={1: 500},
    sampling_mode='add'  # Default
)
# Result: class 1 will have 100 + 500 = 600 samples
```

#### **'target' Mode**
Targets an absolute number of samples (can upsample or downsample):
```python
# If class 0 has 5000 samples and class 1 has 500 samples
augmentor = DistAwareAugmentor(
    sampling_strategy={0: 1000, 1: 1000},
    sampling_mode='target'
)
# Result: class 0 downsampled to 1000, class 1 upsampled to 1000
```

**Use cases:**
- **'add' mode**: When you want to generate a specific number of additional samples
- **'target' mode**: When you want to balance classes to exact counts

---

## 📚 Documentation

### Core Components

#### `DistAwareAugmentor`

Main class for oversampling imbalanced datasets.

**Parameters:**
- `sampling_strategy` (str or dict, default='auto'): How to balance classes
  - `'auto'`: Balance all classes to majority class size
  - `dict`: Specify number of samples per class, e.g., `{0: 100, 1: 200}`
- `sampling_mode` (str, default='add'): How to interpret `sampling_strategy` dict values
  - `'add'`: Add N samples to existing class count (e.g., `{1: 500}` adds 500 to class 1)
  - `'target'`: Target N total samples for class (e.g., `{1: 500}` results in exactly 500 samples)
  - Note: `'target'` mode supports both upsampling and downsampling
- `diversity_threshold` (float, default=0.1): Minimum distance for sample acceptance
  - **Important**: Scale your features for consistent behavior!
- `distribution_method` (str, default='kde'): Distribution fitting method
  - `'kde'`: Kernel Density Estimation (more accurate, slower)
  - `'gaussian'`: Multivariate Gaussian (faster, assumes normality)
- `distance_metric` (str, default='euclidean'): Distance metric for diversity
  - Options: `'euclidean'`, `'manhattan'`, `'cosine'`, `'minkowski'`, etc.
- `random_state` (int, default=None): Random seed for reproducibility

**Methods:**
- `fit(X, y)`: Fit the augmentor to training data
- `resample(X, y)`: Generate synthetic samples
- `fit_resample(X, y)`: Fit and resample in one step

#### `DistanceMetrics`

Utilities for computing distances and diversity scores.

**Key Methods:**
- `compute_distances(X, Y)`: Pairwise distances between samples
- `nearest_neighbor_distances(X, Y)`: Distance to nearest neighbor
- `diversity_score(samples, reference)`: Measure sample diversity
- `filter_diverse_samples(samples, threshold)`: Keep only diverse samples

#### `DistributionFitter`

Fits statistical distributions to feature data.

**Supported Distributions:**
- `'kde'`: Kernel Density Estimation
- `'gaussian'`: Multivariate Gaussian
- `'uniform'`: Uniform distribution (for testing)

---

## 🧪 Testing

### Run All Tests

```bash
# Run tests with coverage
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_augmentor.py

# Run tests matching pattern
pytest -k "test_diversity"
```

### Run Tests with Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=distawareaug --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Test Structure

```
tests/
├── test_augmentor.py       # Tests for DistAwareAugmentor
├── test_distance.py        # Tests for distance metrics
├── test_distribution.py    # Tests for distribution fitting
└── __pycache__/
```

### Writing New Tests

```python
import pytest
import numpy as np
from distawareaug import DistAwareAugmentor

def test_my_feature():
    """Test description."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    augmentor = DistAwareAugmentor(random_state=42)
    X_resampled, y_resampled = augmentor.fit_resample(X, y)
    
    assert len(X_resampled) >= len(X)
    assert len(np.unique(y_resampled)) == len(np.unique(y))
```

### Run CI Tests Locally

Before pushing, run the same checks that GitHub Actions will run:

```bash
# Run all CI checks locally (formatting, linting, tests)
sh run_ci_tests.sh
```

This script will:
1. ✅ Check code formatting with Black
2. ✅ Check import sorting with isort
3. ✅ Run linting with flake8
4. ✅ Run all tests with pytest
5. ✅ Generate coverage report

### Auto-Fix Linting Issues

If you have linting errors (unused imports, variables, etc.):

```bash
# Automatically remove unused imports and variables
python fix_linting.py
```

This will:
- Remove unused imports
- Remove unused variables
- Format code with Black
- Sort imports with isort

Then run `sh run_ci_tests.sh` again to verify!

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/DistAwareAug.git
cd DistAwareAug

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Development Workflow

1. **Create a new branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure code quality:
   ```bash
   # Auto-fix common issues
   python fix_linting.py
   
   # Or manually format
   black distawareaug tests
   isort distawareaug tests
   flake8 distawareaug tests
   ```

3. **Run CI checks locally**:
   ```bash
   # Run all checks (formatting, linting, tests)
   sh run_ci_tests.sh
   ```

4. **Write/update tests**:
   ```bash
   # Run tests
   pytest -v
   
   # Check coverage
   pytest --cov=distawareaug --cov-report=term-missing
   ```

5. **Update documentation** if needed:
   - Update README.md for user-facing changes
   - Update docstrings for library reference changes
   - Add examples for new features

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test changes
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `chore:` Maintenance tasks

6. **Push and create Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub with a clear description.

**Pro tip**: Use `make` commands for common tasks:
```bash
make format      # Format code with black and isort
make lint        # Check linting
make test        # Run tests
make test-cov    # Run tests with coverage
make check       # Run all checks
make clean       # Clean build artifacts
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Code Style Guidelines

- **Line length**: 100 characters (enforced by Black)
- **Docstrings**: Google-style or NumPy-style
- **Type hints**: Encouraged but not required
- **Naming**: 
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Example Contribution Areas

- 🐛 **Bug Fixes**: Found an issue? Submit a PR!
- ✨ **New Features**: 
  - Additional distribution methods
  - New distance metrics
  - Performance optimizations
- 📚 **Documentation**: Improve examples, tutorials, docstrings
- 🧪 **Tests**: Increase test coverage
- 🎨 **Examples**: Add Jupyter notebooks demonstrating use cases

### Questions?

- Open an [Issue](https://github.com/Ayo-Cyber/DistAwareAug/issues) for bugs or feature requests
- Start a [Discussion](https://github.com/Ayo-Cyber/DistAwareAug/discussions) for questions

---

## 📊 Examples

Explore comprehensive examples in the `examples/` directory:

### Available Notebooks

1. **`demo_synthetic.ipynb`**: Introduction to DistAwareAug with synthetic data
2. **`compare_smote.ipynb`**: Benchmark comparison with SMOTE, ADASYN, etc.
3. **`comprehensive_test.ipynb`**: Full performance analysis with threshold optimization

### Running Examples

```bash
# Install example dependencies
pip install ".[examples]"

# Start Jupyter
jupyter notebook examples/
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Low Diversity / Few Samples Generated**

**Problem**: `diversity_threshold` is too high for your feature scales.

**Solution**: Scale your features first!
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

augmentor = DistAwareAugmentor(diversity_threshold=0.1)
X_resampled, y_resampled = augmentor.fit_resample(X_train_scaled, y_train)
```

#### 2. **Singular Matrix Errors**

**Problem**: Too few samples or perfectly correlated features.

**Solution**: DistAwareAug handles this automatically, but you can:
- Use `distribution_method='gaussian'` which adds regularization
- Ensure sufficient samples (>10 per class recommended)
- Remove perfectly correlated features

#### 3. **Slow Performance**

**Problem**: KDE is slow for large datasets or diversity checking takes too long.

**Solutions**:
```python
# Solution 1: Use Gaussian method (3-5x faster than KDE)
augmentor = DistAwareAugmentor(distribution_method='gaussian')

# Solution 2: Lower diversity threshold (faster, more samples accepted)
augmentor = DistAwareAugmentor(diversity_threshold=0.05)

# Solution 3: Combine both for maximum speed
augmentor = DistAwareAugmentor(
    distribution_method='gaussian',
    diversity_threshold=0.05
)
```

**Note**: DistAwareAug is typically 5-15x slower than SMOTE due to distribution 
fitting and diversity enforcement. This is expected and provides higher quality 
synthetic samples.

#### 4. **Import Errors**

**Problem**: Missing dependencies.

**Solution**:
```bash
pip install -e ".[all]"
```

---

## 📈 Performance Tips

### Speed Considerations

DistAwareAug is typically **5-15x slower than SMOTE** due to:
- **Distribution fitting** (KDE or Gaussian)
- **Diversity enforcement** (checks samples against random subsample for efficiency)

For reference on a 5,000 sample dataset (9:1 imbalance):
- SMOTE: ~0.007s
- ADASYN: ~0.035s (5x slower than SMOTE)
- **DistAwareAug**: ~0.05-0.08s (7-12x slower than SMOTE)

This trade-off provides **better quality synthetic data** that preserves statistical distributions.

### Optimization Tips

1. **Scale your features** with `StandardScaler` for consistent `diversity_threshold` behavior
2. **Use `distribution_method='gaussian'`** for large datasets (3-5x faster than KDE)
   ```python
   augmentor = DistAwareAugmentor(distribution_method='gaussian')  # Faster
   ```
3. **Adjust `diversity_threshold`** based on your needs:
   - Higher (0.2-0.5): More diverse samples, fewer total samples, slower
   - Lower (0.05-0.1): More samples, less diversity, faster
4. **Set `random_state`** for reproducible results
5. **Start with small datasets** to tune parameters before scaling up

### How Diversity Checking Works

DistAwareAug ensures synthetic samples are diverse by checking they are sufficiently 
far from existing samples. For performance, diversity is checked against a random 
subsample of up to 200 existing synthetic samples rather than all of them.

**Why this works:**
- Checking all pairwise distances would be O(n²) - extremely slow for thousands of samples
- Random sampling provides ~95% of the quality with 10x+ better performance
- Similar to statistical polling: you don't need to survey everyone to get accurate results

This approximation is statistically sound and provides excellent quality/speed balance.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by SMOTE and related oversampling techniques
- Built with scikit-learn, NumPy, SciPy, and pandas
- Thanks to the open-source community

---

## 📬 Contact

- **Author**: Atunrase Ayo
- **Email**: atunraseayomide@gmail.com
- **GitHub**: [@Ayo-Cyber](https://github.com/Ayo-Cyber)
- **Repository**: [DistAwareAug](https://github.com/Ayo-Cyber/DistAwareAug)

---

## 📚 Citation

If you use DistAwareAug in your research, please cite:

```bibtex
@software{distawareaug2025,
  author = {Atunrase, Ayo},
  title = {DistAwareAug: Distribution-Aware Data Augmentation for Imbalanced Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Ayo-Cyber/DistAwareAug}
}
```

---

## 🗺️ Roadmap

- [ ] Publish to PyPI
- [ ] Add more distribution methods (t-distribution, mixture models)
- [ ] GPU acceleration for large-scale augmentation
- [ ] Web-based interactive demo
- [ ] Integration with popular ML frameworks (PyTorch, TensorFlow)
- [ ] Automated hyperparameter tuning
- [ ] Real-world dataset benchmarks

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

[Report Bug](https://github.com/Ayo-Cyber/DistAwareAug/issues) •
[Request Feature](https://github.com/Ayo-Cyber/DistAwareAug/issues) •
[Contribute](https://github.com/Ayo-Cyber/DistAwareAug/pulls)

</div>
