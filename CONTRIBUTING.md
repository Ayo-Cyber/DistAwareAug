# Contributing to DistAwareAug

Thank you for your interest in contributing to DistAwareAug! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility and apologize to those affected by our mistakes

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DistAwareAug.git
   cd DistAwareAug
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Ayo-Cyber/DistAwareAug.git
   ```

4. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

5. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

6. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 🔨 How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version, OS, package version
   - Minimal code example if applicable

### Suggesting Features

1. **Check existing feature requests**
2. **Create a new issue** with:
   - Clear description of the feature
   - Use cases and motivation
   - Possible implementation approach (optional)
   - Examples of how it would be used

### Submitting Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**:
   - Write clear, readable code
   - Follow coding standards (see below)
   - Add/update tests
   - Update documentation

3. **Test your changes**:
   ```bash
   pytest -v
   pytest --cov=distawareaug
   ```

4. **Format your code**:
   ```bash
   black distawareaug tests
   isort distawareaug tests
   flake8 distawareaug tests
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "type: brief description"
   ```
   
   Use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation only
   - `test:` Adding or updating tests
   - `refactor:` Code change that neither fixes a bug nor adds a feature
   - `perf:` Performance improvement
   - `chore:` Maintenance tasks
   - `ci:` CI/CD changes

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request** on GitHub

## 📝 Coding Standards

### Python Style

- **PEP 8** compliant (enforced by flake8)
- **Line length**: 100 characters max (enforced by black)
- **Imports**: Organized with isort
  ```python
  # Standard library
  import os
  import sys
  
  # Third-party
  import numpy as np
  from sklearn.base import BaseEstimator
  
  # Local
  from .utils import validate_data
  ```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`

### Documentation

Use **Google-style** or **NumPy-style** docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    Longer description if needed. Explain what the function does,
    not just how.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Examples
    --------
    >>> function_name(42, "hello")
    True
    """
    pass
```

### Type Hints

Encouraged but not required:

```python
from typing import Optional, Union, Tuple

def process_data(
    X: np.ndarray, 
    y: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Process data with type hints."""
    pass
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- One test file per source file
- Use descriptive test names

```python
import pytest
import numpy as np
from distawareaug import DistAwareAugmentor

def test_augmentor_creates_more_samples():
    """Test that augmentor increases the number of samples."""
    X = np.random.randn(100, 5)
    y = np.array([0] * 80 + [1] * 20)
    
    augmentor = DistAwareAugmentor(random_state=42)
    X_resampled, y_resampled = augmentor.fit_resample(X, y)
    
    assert len(X_resampled) > len(X)
    assert len(X_resampled) == len(y_resampled)
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Test both success and failure paths

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=distawareaug --cov-report=html

# Run specific test file
pytest tests/test_augmentor.py

# Run specific test
pytest tests/test_augmentor.py::test_augmentor_creates_more_samples

# Run tests matching pattern
pytest -k "diversity"
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### PR Description

Include:

1. **Summary**: What does this PR do?
2. **Motivation**: Why is this change needed?
3. **Changes**: List of key changes
4. **Testing**: How was it tested?
5. **Screenshots**: If UI changes (N/A for this project)
6. **Related Issues**: Fixes #123, Closes #456

### Review Process

1. Automated checks must pass (tests, linting)
2. At least one maintainer review required
3. Address review comments
4. Maintainer will merge when approved

### After Merge

1. Delete your branch
2. Pull latest changes:
   ```bash
   git checkout main
   git pull upstream main
   ```

## 📦 Release Process

(For maintainers)

1. **Update version** in `pyproject.toml` and `__init__.py`
2. **Update CHANGELOG.md**
3. **Create release tag**:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push upstream v0.1.0
   ```
4. **Build and publish** to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```

## 💡 Tips for Contributors

### Good First Issues

Look for issues labeled:
- `good first issue`
- `beginner-friendly`
- `documentation`
- `help wanted`

### Communication

- **Be patient**: Maintainers are often volunteers
- **Be respectful**: Assume good intentions
- **Be clear**: Provide context and details
- **Be collaborative**: We're all learning together

### Getting Help

- Comment on the issue you're working on
- Ask questions in discussions
- Reach out to maintainers if stuck

## 🎯 Areas Looking for Contributions

We especially welcome contributions in:

1. **Documentation**:
   - Tutorials and examples
   - API documentation improvements
   - Typo fixes and clarifications

2. **Testing**:
   - Increase test coverage
   - Add edge case tests
   - Performance benchmarks

3. **Features**:
   - New distribution methods
   - Additional distance metrics
   - Performance optimizations

4. **Bug Fixes**:
   - Check open issues
   - Report and fix bugs you encounter

5. **Examples**:
   - Real-world use cases
   - Jupyter notebooks
   - Comparison studies

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make DistAwareAug better for everyone. We appreciate your time and effort!

---

**Questions?** Open an issue or discussion on GitHub.
