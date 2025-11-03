# Changelog

All notable changes to DistAwareAug will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-03

### Added
- **Flexible Sampling Modes**: New `sampling_mode` parameter with 'add' and 'target' options
  - `'add'` mode: Add N samples to existing class count (default, backward compatible)
  - `'target'` mode: Target absolute sample counts with support for both upsampling and downsampling
- Downsampling support when using `sampling_mode='target'`
- Comprehensive test notebook (`comprehensive_test.ipynb`) with threshold optimization
- Statistical subsampling for diversity checks (max 200 random samples)

### Changed
- **Performance Improvements**: 10-13x speedup through multiple optimizations
  - Added `n_jobs=-1` to NearestNeighbors for parallel processing
  - Increased batch size from 500 to 2000 for fewer Python-level iterations
  - Vectorized feature clipping using `np.clip` instead of per-sample loops
  - Implemented statistical subsampling for diversity checks (O(n) instead of O(n²))
  - Replaced repeated NearestNeighbors fits with vectorized `pairwise_distances`
- Reused `RandomState` object throughout instead of creating new instances
- Overall performance: from 91x slower than SMOTE to 7-10x slower (acceptable trade-off for quality)

### Fixed
- Data ordering preservation in `resample()` when no downsampling occurs
- Resolved test failures related to data reordering
- Improved robustness of diversity checking for large-scale augmentation

### Documentation
- Added detailed `sampling_mode` parameter documentation
- Added performance benchmarks and comparisons with SMOTE/ADASYN
- Explained diversity checking implementation with statistical justification
- Added "Understanding Sampling Modes" section with clear examples
- Updated troubleshooting guide with performance optimization tips
- Enhanced README with real-world performance numbers

## [0.1.0] - 2025-10-22

### Added
- First alpha release of DistAwareAug
- Core functionality for distribution-aware augmentation
- Support for imbalanced dataset oversampling
- KDE and Gaussian distribution fitting
- Multiple distance metrics (Euclidean, Manhattan, Cosine, etc.)
- Diversity checking and filtering
- scikit-learn compatible API
- Comprehensive documentation
- Unit tests with pytest
- Example notebooks

[Unreleased]: https://github.com/Ayo-Cyber/DistAwareAug/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Ayo-Cyber/DistAwareAug/releases/tag/v0.2.0
[0.1.0]: https://github.com/Ayo-Cyber/DistAwareAug/releases/tag/v0.1.0
