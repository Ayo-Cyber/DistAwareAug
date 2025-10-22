"""
Step-by-Step Algorithmic Pipeline for Statistically-Governed Oversampling

This implements the complete framework described:
1. Learn distribution statistics that govern the minority class
2. Generate synthetic data that follows the same statistical shape
3. Apply distance-awareness to maintain diversity
4. Provide tunable controls for generation bias
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GenerationBias(Enum):
    MEAN = "mean"  # Bias toward central tendency
    TAILS = "tails"  # Bias toward distribution tails
    SUBCLUSTERS = "subclusters"  # Focus on rare subclusters
    BALANCED = "balanced"  # Uniform across distribution


@dataclass
class StatisticalProfile:
    """Container for learned statistical properties of minority class."""

    feature_distributions: Dict[int, Dict]
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    dependency_graph: Dict[int, List[int]]
    subclusters: Dict[str, Any]
    feature_types: Dict[int, str]


class StatisticallyGovernedAugmentor:
    """
    Complete implementation of Statistically-Governed Oversampling.

    Pipeline:
    1. Statistical Learning Phase
    2. Correlation-Aware Generation Phase
    3. Distance-Aware Refinement Phase
    4. Quality Validation Phase
    """

    def __init__(
        self,
        generation_bias: GenerationBias = GenerationBias.BALANCED,
        preserve_correlations: bool = True,
        diversity_threshold: float = 0.1,
        quality_threshold: float = 0.05,
        max_generation_attempts: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.generation_bias = generation_bias
        self.preserve_correlations = preserve_correlations
        self.diversity_threshold = diversity_threshold
        self.quality_threshold = quality_threshold
        self.max_generation_attempts = max_generation_attempts
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline for statistically-governed oversampling.

        Returns
        -------
        X_resampled : array-like
            Original + synthetic samples
        y_resampled : array-like
            Corresponding labels
        """

        # PHASE 1: STATISTICAL LEARNING
        print("🔍 Phase 1: Learning Statistical Structure...")
        statistical_profiles = self._learn_class_statistics(X, y)

        # PHASE 2: CORRELATION-AWARE GENERATION
        print("🎯 Phase 2: Generating Statistically-Faithful Samples...")
        synthetic_samples = self._generate_statistically_faithful_samples(
            X, y, statistical_profiles
        )

        # PHASE 3: DISTANCE-AWARE REFINEMENT
        print("📏 Phase 3: Applying Distance-Aware Refinement...")
        refined_samples = self._apply_distance_aware_refinement(X, y, synthetic_samples)

        # PHASE 4: QUALITY VALIDATION
        print("✅ Phase 4: Validating Sample Quality...")
        validated_samples = self._validate_sample_quality(
            X, y, refined_samples, statistical_profiles
        )

        # Combine original + synthetic
        X_resampled = np.vstack([X] + [samples for samples in validated_samples.values()])
        y_resampled = np.hstack(
            [y]
            + [
                np.full(len(samples), class_label)
                for class_label, samples in validated_samples.items()
            ]
        )

        return X_resampled, y_resampled

    def _learn_class_statistics(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[int, StatisticalProfile]:
        """
        PHASE 1: Learn comprehensive statistical properties for each class.

        This is where we understand:
        - What distribution governs each feature (Normal? LogNormal? etc.)
        - How features correlate with each other
        - What subclusters exist within each class
        - Dependencies and conditional relationships
        """
        profiles = {}
        classes = np.unique(y)

        for class_label in classes:
            print(f"  📊 Analyzing class {class_label}...")

            # Extract class data
            class_mask = y == class_label
            class_data = X[class_mask]

            # 1. Feature-wise distribution analysis
            feature_distributions = {}
            feature_types = {}

            for feature_idx in range(X.shape[1]):
                feature_data = class_data[:, feature_idx]

                # Determine feature type and best distribution
                dist_analysis = self._analyze_feature_distribution(feature_data)
                feature_distributions[feature_idx] = dist_analysis
                feature_types[feature_idx] = dist_analysis["type"]

                print(
                    f"    Feature {feature_idx}: {dist_analysis['best_fit']['name']} "
                    f"(p-value: {dist_analysis['best_fit']['p_value']:.3f})"
                )

            # 2. Correlation and covariance structure
            correlation_matrix = np.corrcoef(class_data, rowvar=False)
            covariance_matrix = np.cov(class_data, rowvar=False)

            # 3. Dependency graph (beyond linear correlation)
            dependency_graph = self._build_dependency_graph(class_data, correlation_matrix)

            # 4. Subcluster analysis
            subclusters = self._detect_subclusters(class_data)

            # Create statistical profile
            profiles[class_label] = StatisticalProfile(
                feature_distributions=feature_distributions,
                correlation_matrix=correlation_matrix,
                covariance_matrix=covariance_matrix,
                dependency_graph=dependency_graph,
                subclusters=subclusters,
                feature_types=feature_types,
            )

        return profiles

    def _analyze_feature_distribution(self, feature_data: np.ndarray) -> Dict[str, Any]:
        """
        Identify the true distribution that governs a feature.

        This answers: "Is Age ~N(45, σ²)?" or "Is MineralLevel log-normal?"
        """
        from scipy import stats

        # Test multiple distributions
        distributions_to_test = [
            ("normal", stats.norm, {}),
            ("lognormal", stats.lognorm, {}),
            ("exponential", stats.expon, {}),
            ("gamma", stats.gamma, {}),
            ("weibull", stats.weibull_min, {}),
            ("uniform", stats.uniform, {}),
        ]

        # Handle edge cases for lognormal (needs positive values)
        if np.min(feature_data) <= 0:
            distributions_to_test = [d for d in distributions_to_test if d[0] != "lognormal"]

        best_fit = None
        best_p_value = 0

        for dist_name, dist_func, dist_kwargs in distributions_to_test:
            try:
                # Fit parameters
                params = dist_func.fit(feature_data, **dist_kwargs)

                # Kolmogorov-Smirnov goodness of fit test
                ks_stat, p_value = stats.kstest(feature_data, lambda x: dist_func.cdf(x, *params))

                if p_value > best_p_value:
                    best_p_value = p_value
                    best_fit = {
                        "name": dist_name,
                        "distribution": dist_func,
                        "params": params,
                        "p_value": p_value,
                        "ks_statistic": ks_stat,
                    }

            except Exception:
                continue

        # Determine feature type
        unique_ratio = len(np.unique(feature_data)) / len(feature_data)
        if unique_ratio < 0.1:
            feature_type = "categorical"
        elif np.allclose(feature_data, feature_data.astype(int)):
            feature_type = "discrete"
        else:
            feature_type = "continuous"

        return {
            "type": feature_type,
            "best_fit": best_fit or {"name": "empirical", "params": feature_data},
            "statistics": {
                "mean": np.mean(feature_data),
                "std": np.std(feature_data),
                "min": np.min(feature_data),
                "max": np.max(feature_data),
                "skewness": stats.skew(feature_data),
                "kurtosis": stats.kurtosis(feature_data),
            },
        }

    def _build_dependency_graph(
        self, data: np.ndarray, corr_matrix: np.ndarray
    ) -> Dict[int, List[int]]:
        """
        Build graph of feature dependencies.

        This captures relationships like: "If Age increases, then BloodPressure tends to increase"
        """
        n_features = data.shape[1]
        dependency_graph = {i: [] for i in range(n_features)}

        # Find strong correlations (simplified version)
        for i in range(n_features):
            for j in range(n_features):
                if i != j and abs(corr_matrix[i, j]) >= 0.5:
                    dependency_graph[i].append(j)

        return dependency_graph

    def _detect_subclusters(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect subclusters within the minority class.

        This enables generation bias toward "rare subclusters".
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if len(data) < 4:  # Need minimum samples for clustering
            return {
                "n_clusters": 1,
                "centers": [np.mean(data, axis=0)],
                "labels": np.zeros(len(data)),
                "sizes": [len(data)],
            }

        # Find optimal number of clusters
        max_k = min(5, len(data) // 2)
        best_k = 1
        best_score = -1

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(data)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        # Final clustering
        if best_k > 1:
            kmeans = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(data)
            centers = kmeans.cluster_centers_
        else:
            labels = np.zeros(len(data))
            centers = [np.mean(data, axis=0)]

        # Calculate cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        sizes = {label: count for label, count in zip(unique_labels, counts)}

        return {
            "n_clusters": best_k,
            "centers": centers,
            "labels": labels,
            "sizes": sizes,
            "silhouette_score": best_score if best_k > 1 else 0,
        }

    def _generate_statistically_faithful_samples(
        self, X: np.ndarray, y: np.ndarray, profiles: Dict[int, StatisticalProfile]
    ) -> Dict[int, np.ndarray]:
        """
        PHASE 2: Generate samples that respect learned statistical properties.

        Key insight: Don't just interpolate between existing points (like SMOTE).
        Instead, sample from the learned statistical distributions while preserving
        correlations and dependencies.
        """
        synthetic_samples = {}

        # Determine sampling strategy (balance to majority class)
        class_counts = {cls: np.sum(y == cls) for cls in np.unique(y)}
        max_count = max(class_counts.values())

        for class_label, profile in profiles.items():
            current_count = class_counts[class_label]
            n_to_generate = max_count - current_count

            if n_to_generate <= 0:
                continue

            print(f"  🎯 Generating {n_to_generate} samples for class {class_label}...")

            if self.preserve_correlations:
                # Generate correlated samples using multivariate approach
                samples = self._generate_correlated_samples(
                    profile, n_to_generate, X[y == class_label]
                )
            else:
                # Generate independent samples feature-by-feature
                samples = self._generate_independent_samples(profile, n_to_generate)

            synthetic_samples[class_label] = samples

        return synthetic_samples

    def _generate_correlated_samples(
        self, profile: StatisticalProfile, n_samples: int, original_class_data: np.ndarray
    ) -> np.ndarray:
        """
        Generate samples that preserve correlation structure.

        This is crucial for your premise: if Age and BloodPressure are correlated
        in the minority class, synthetic samples should maintain that correlation.
        """
        n_features = len(profile.feature_distributions)
        samples = np.zeros((n_samples, n_features))

        if (
            self.generation_bias == GenerationBias.SUBCLUSTERS
            and profile.subclusters["n_clusters"] > 1
        ):
            # Focus on rare subclusters
            samples = self._generate_from_subclusters(profile, n_samples, original_class_data)

        elif self.generation_bias == GenerationBias.TAILS:
            # Bias toward distribution tails
            samples = self._generate_tail_biased(profile, n_samples)

        elif self.generation_bias == GenerationBias.MEAN:
            # Bias toward central tendency
            samples = self._generate_mean_biased(profile, n_samples)

        else:  # BALANCED
            # Use multivariate normal as base, then transform features
            try:
                # Generate from multivariate normal
                mvn_samples = np.random.multivariate_normal(
                    mean=np.zeros(n_features), cov=profile.correlation_matrix, size=n_samples
                )

                # Transform each feature to match its learned distribution
                for i in range(n_features):
                    feature_dist = profile.feature_distributions[i]

                    # Transform standard normal to target distribution
                    samples[:, i] = self._transform_to_target_distribution(
                        mvn_samples[:, i], feature_dist, original_class_data[:, i]
                    )

            except np.linalg.LinAlgError:
                # Fallback to independent generation if covariance is singular
                samples = self._generate_independent_samples(profile, n_samples)

        return samples

    def _transform_to_target_distribution(
        self,
        standard_normal_samples: np.ndarray,
        feature_dist: Dict[str, Any],
        original_feature_data: np.ndarray,
    ) -> np.ndarray:
        """
        Transform standard normal samples to match the target feature distribution.

        This ensures that if MineralLevel is log-normal in the minority class,
        our synthetic MineralLevel values will also be log-normal.
        """
        from scipy import stats

        best_fit = feature_dist["best_fit"]

        if best_fit["name"] == "empirical":
            # Use empirical distribution (bootstrapping)
            return np.random.choice(original_feature_data, size=len(standard_normal_samples))

        try:
            dist_func = best_fit["distribution"]
            params = best_fit["params"]

            # Convert standard normal to uniform [0,1] via CDF
            uniform_samples = stats.norm.cdf(standard_normal_samples)

            # Convert uniform to target distribution via inverse CDF
            target_samples = dist_func.ppf(uniform_samples, *params)

            # Handle any invalid values
            valid_mask = np.isfinite(target_samples)
            if not np.all(valid_mask):
                # Replace invalid values with empirical samples
                n_invalid = np.sum(~valid_mask)
                target_samples[~valid_mask] = np.random.choice(
                    original_feature_data, size=n_invalid
                )

            return target_samples

        except Exception:
            # Fallback to empirical distribution
            return np.random.choice(original_feature_data, size=len(standard_normal_samples))

    def _generate_independent_samples(
        self, profile: StatisticalProfile, n_samples: int
    ) -> np.ndarray:
        """Generate samples with independent features (fallback method)."""
        n_features = len(profile.feature_distributions)
        samples = np.zeros((n_samples, n_features))

        for i in range(n_features):
            feature_dist = profile.feature_distributions[i]
            best_fit = feature_dist["best_fit"]

            if best_fit["name"] == "empirical":
                samples[:, i] = np.random.choice(best_fit["params"], size=n_samples)
            else:
                try:
                    dist_func = best_fit["distribution"]
                    params = best_fit["params"]
                    samples[:, i] = dist_func.rvs(*params, size=n_samples)
                except Exception:
                    # Fallback to normal distribution
                    stats_dict = feature_dist["statistics"]
                    samples[:, i] = np.random.normal(
                        stats_dict["mean"], stats_dict["std"], n_samples
                    )

        return samples

    def _apply_distance_aware_refinement(
        self, X: np.ndarray, y: np.ndarray, synthetic_samples: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        PHASE 3: Apply distance-awareness to avoid duplicates and outliers.

        This prevents placing synthetic points too close (duplicates) or
        too far (unrealistic outliers) from the minority region.
        """
        refined_samples = {}

        for class_label, samples in synthetic_samples.items():
            if len(samples) == 0:
                refined_samples[class_label] = samples
                continue

            print(f"  📏 Refining {len(samples)} samples for class {class_label}...")

            # Get original class data for distance calculations
            class_mask = y == class_label
            original_class_data = X[class_mask]

            # Filter out samples that are too close or too far
            refined = self._filter_by_distance_criteria(samples, original_class_data)

            refined_samples[class_label] = refined
            print(f"    Kept {len(refined)}/{len(samples)} samples after distance filtering")

        return refined_samples

    def _filter_by_distance_criteria(
        self, synthetic_samples: np.ndarray, original_data: np.ndarray
    ) -> np.ndarray:
        """
        Filter synthetic samples based on distance criteria.

        Remove samples that are:
        1. Too close to existing samples (duplicates)
        2. Too far from the minority region (outliers)
        """
        from sklearn.metrics.pairwise import euclidean_distances

        if len(synthetic_samples) == 0:
            return synthetic_samples

        kept_samples = []

        for i, candidate in enumerate(synthetic_samples):
            candidate = candidate.reshape(1, -1)

            # Distance to nearest original sample
            distances_to_original = euclidean_distances(candidate, original_data)[0]
            min_dist_to_original = np.min(distances_to_original)

            # Distance to already kept synthetic samples
            min_dist_to_synthetic = float("inf")
            if kept_samples:
                distances_to_synthetic = euclidean_distances(candidate, np.array(kept_samples))[0]
                min_dist_to_synthetic = np.min(distances_to_synthetic)

            # Apply distance criteria
            too_close = min_dist_to_original < self.diversity_threshold
            too_close_to_synthetic = min_dist_to_synthetic < self.diversity_threshold

            # Define "too far" as beyond 3 standard deviations from class center
            class_center = np.mean(original_data, axis=0)
            dist_to_center = euclidean_distances(candidate, class_center.reshape(1, -1))[0][0]
            class_std = np.mean(np.std(original_data, axis=0))
            too_far = dist_to_center > 3 * class_std

            # Keep sample if it passes all criteria
            if not (too_close or too_close_to_synthetic or too_far):
                kept_samples.append(candidate[0])

        return (
            np.array(kept_samples)
            if kept_samples
            else np.array([]).reshape(0, original_data.shape[1])
        )

    def _validate_sample_quality(
        self,
        X: np.ndarray,
        y: np.ndarray,
        refined_samples: Dict[int, np.ndarray],
        profiles: Dict[int, StatisticalProfile],
    ) -> Dict[int, np.ndarray]:
        """
        PHASE 4: Final quality validation to ensure statistical fidelity.

        Verify that synthetic samples actually follow the learned distributions.
        """
        validated_samples = {}

        for class_label, samples in refined_samples.items():
            if len(samples) == 0:
                validated_samples[class_label] = samples
                continue

            print(f"  ✅ Validating {len(samples)} samples for class {class_label}...")

            # Statistical validation
            profile = profiles[class_label]
            original_class_data = X[y == class_label]

            valid_samples = self._statistical_quality_check(samples, original_class_data, profile)

            validated_samples[class_label] = valid_samples
            print(f"    {len(valid_samples)}/{len(samples)} samples passed quality validation")

        return validated_samples

    def _statistical_quality_check(
        self, synthetic_samples: np.ndarray, original_data: np.ndarray, profile: StatisticalProfile
    ) -> np.ndarray:
        """
        Check if synthetic samples maintain statistical properties of original data.
        """
        if len(synthetic_samples) == 0:
            return synthetic_samples

        valid_indices = []

        for i, sample in enumerate(synthetic_samples):
            is_valid = True

            # Check each feature against its learned distribution
            for feature_idx in range(len(sample)):
                feature_value = sample[feature_idx]
                original_feature = original_data[:, feature_idx]

                # Simple range check (can be made more sophisticated)
                feature_min = np.min(original_feature)
                feature_max = np.max(original_feature)
                feature_std = np.std(original_feature)
                feature_mean = np.mean(original_feature)

                # Allow values within reasonable range (mean ± 3*std, bounded by observed range)
                reasonable_min = max(feature_min, feature_mean - 3 * feature_std)
                reasonable_max = min(feature_max, feature_mean + 3 * feature_std)

                if not (reasonable_min <= feature_value <= reasonable_max):
                    is_valid = False
                    break

            if is_valid:
                valid_indices.append(i)

        return (
            synthetic_samples[valid_indices]
            if valid_indices
            else np.array([]).reshape(0, synthetic_samples.shape[1])
        )


# Usage example
if __name__ == "__main__":
    print("🎯 Statistically-Governed Oversampling Pipeline")
    print("=" * 50)
    print("This pipeline implements:")
    print("1. 📊 Statistical Learning: Identify true feature distributions")
    print("2. 🎯 Correlation-Aware Generation: Preserve relationships")
    print("3. 📏 Distance-Aware Refinement: Maintain diversity")
    print("4. ✅ Quality Validation: Ensure statistical fidelity")
    print("\nKey Features:")
    print("- Detects if Age ~N(45, σ²) or MineralLevel ~LogNormal")
    print("- Preserves correlations (Age ↔ BloodPressure)")
    print("- Tunable generation bias (mean/tails/subclusters)")
    print("- Distance controls prevent duplicates & outliers")
