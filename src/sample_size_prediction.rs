//! # AAC Sample Size Predictor
//!
//! A library for predicting original AAC frame sizes from AES-128 CBC encrypted,
//! 16-byte aligned sample data. Designed for MP4 processing and moov atom size estimation.
//!
//! ## Features
//!
//! - Pattern learning from encrypted/original size pairs
//! - Sequential context analysis for improved accuracy
//! - AAC VBR domain-specific optimizations
//! - Confidence scoring and validation
//! - Optimized for 16-byte aligned AES encryption
//!
//! ## Example
//!
//! ```rust
//! use aac_predictor::{AacSamplePredictor, AudioProfile};
//!
//! let mut predictor = AacSamplePredictor::new();
//!
//! // Train on known data
//! let encrypted_sizes = vec![368, 352, 336];
//! let original_sizes = vec![361, 349, 321];
//! predictor.train(&encrypted_sizes, &original_sizes)?;
//!
//! // Predict unknown samples
//! let test_encrypted = vec![384, 368, 352];
//! let results = predictor.predict(&test_encrypted, 0);
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// Errors that can occur during prediction operations
#[derive(Debug, thiserror::Error)]
pub enum PredictionError {
    #[error("Invalid AES padding: {padding} bytes (encrypted: {encrypted}, original: {original})")]
    InvalidPadding {
        padding: u32,
        encrypted: u32,
        original: u32,
    },

    #[error("Encrypted size {size} not 16-byte aligned")]
    NotAligned { size: u32 },

    #[error("AAC frame size {size} outside expected range (100-4000 bytes)")]
    InvalidAacSize { size: u32 },

    #[error("Training data length mismatch: encrypted={encrypted_len}, original={original_len}")]
    DataLengthMismatch {
        encrypted_len: usize,
        original_len: usize,
    },

    #[error("No training data available")]
    NoTrainingData,
}

/// Audio profile characteristics for better prediction accuracy
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct AudioProfile {
    pub sample_rate: u32,
    pub channels: u8,
    pub estimated_duration_ms: u32,
    pub has_chapters: bool,
    pub metadata_complexity: MetadataComplexity,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MetadataComplexity {
    Minimal,
    Standard,
    Rich,
}

/// Bitrate and padding analysis for encrypted sizes
#[derive(Debug, Clone)]
struct BitrateProfile {
    avg_original_size: f64,
    size_variance: f64,
    common_padding_values: Vec<u32>,
    sample_count: usize,
}

/// Main predictor for AAC sample sizes
#[derive(Debug, Clone)]
pub struct AacSamplePredictor {
    // Core pattern storage
    size_patterns: HashMap<u32, Vec<u32>>,
    transition_patterns: HashMap<i32, Vec<i32>>,
    pattern_confidence: HashMap<u32, f64>,

    // AAC-specific analysis
    bitrate_profiles: HashMap<u32, BitrateProfile>,
    frame_size_distribution: Vec<u32>,

    // Statistics
    training_samples: usize,
    is_trained: bool,
}

/// Result of a prediction operation
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub predicted_original_sizes: Vec<u32>,
    pub confidence_scores: Vec<f64>,
    pub exact_matches: usize,
    pub total_predictions: usize,
    pub processing_time_ms: f64,
    pub average_confidence: f64,
}

/// Validation metrics for prediction accuracy
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    pub exact_matches: usize,
    pub within_1_byte: usize,
    pub within_2_bytes: usize,
    pub within_5_bytes: usize,
    pub total_samples: usize,
    pub average_error: f64,
    pub max_error: u32,
}

impl AudioProfile {
    /// Create a new audio profile
    pub fn new(sample_rate: u32, channels: u8, estimated_duration_ms: u32) -> Self {
        Self {
            sample_rate,
            channels,
            estimated_duration_ms,
            has_chapters: false,
            metadata_complexity: MetadataComplexity::Standard,
        }
    }

    /// Estimate the number of AAC frames for this profile
    pub fn estimated_sample_count(&self) -> u32 {
        // AAC frames are typically 1024 samples, ~23ms at 44.1kHz
        let frame_duration_ms = (1024.0 * 1000.0) / self.sample_rate as f64;
        (self.estimated_duration_ms as f64 / frame_duration_ms).ceil() as u32
    }

    /// Calculate expected moov atom size for this profile
    pub fn estimated_moov_size(&self) -> u32 {
        let sample_count = self.estimated_sample_count();
        let base_overhead = 2048u32; // Basic MP4 headers
        let bytes_per_sample = 12u32; // Sample table entries
        let metadata_overhead = match self.metadata_complexity {
            MetadataComplexity::Minimal => 512,
            MetadataComplexity::Standard => 2048,
            MetadataComplexity::Rich => 8192,
        };

        base_overhead + (sample_count * bytes_per_sample) + metadata_overhead
    }
}

impl Default for AacSamplePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl AacSamplePredictor {
    /// Create a new predictor instance
    pub fn new() -> Self {
        Self {
            size_patterns: HashMap::new(),
            transition_patterns: HashMap::new(),
            pattern_confidence: HashMap::new(),
            bitrate_profiles: HashMap::new(),
            frame_size_distribution: Vec::new(),
            training_samples: 0,
            is_trained: false,
        }
    }

    /// Train the predictor on known encrypted/original size pairs
    pub fn train(
        &mut self,
        encrypted_sizes: &[u32],
        original_sizes: &[u32],
    ) -> Result<(), PredictionError> {
        if encrypted_sizes.len() != original_sizes.len() {
            return Err(PredictionError::DataLengthMismatch {
                encrypted_len: encrypted_sizes.len(),
                original_len: original_sizes.len(),
            });
        }

        if encrypted_sizes.is_empty() {
            return Err(PredictionError::NoTrainingData);
        }

        println!("Training on {} samples...", encrypted_sizes.len());

        // Validate and build size patterns
        for (&encrypted, &original) in encrypted_sizes.iter().zip(original_sizes.iter()) {
            self.validate_sample_pair(encrypted, original)?;

            self.size_patterns
                .entry(encrypted)
                .or_insert_with(Vec::new)
                .push(original);
        }

        // Build transition patterns
        self.build_transition_patterns(encrypted_sizes, original_sizes);

        // Calculate confidence scores
        self.calculate_confidence_scores();

        // AAC-specific analysis
        self.analyze_bitrate_patterns(encrypted_sizes, original_sizes);
        self.build_frame_size_distribution(original_sizes);

        self.training_samples = encrypted_sizes.len();
        self.is_trained = true;

        println!("Training completed:");
        println!("  {} unique encrypted sizes", self.size_patterns.len());
        println!("  {} transition patterns", self.transition_patterns.len());
        println!("  {} bitrate profiles", self.bitrate_profiles.len());

        Ok(())
    }

    /// Predict original sizes for encrypted samples
    pub fn predict(&self, encrypted_sizes: &[u32], start_index: usize) -> PredictionResult {
        let start_time = Instant::now();
        let samples_to_predict = &encrypted_sizes[start_index..];

        if samples_to_predict.is_empty() {
            return PredictionResult::empty();
        }

        let mut predictions = Vec::with_capacity(samples_to_predict.len());
        let mut confidences = Vec::with_capacity(samples_to_predict.len());

        for (i, &encrypted_size) in samples_to_predict.iter().enumerate() {
            let (prediction, confidence) = self.predict_single_sample(
                encrypted_size,
                encrypted_sizes,
                start_index + i,
                &predictions,
            );

            predictions.push(prediction);
            confidences.push(confidence);
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let average_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;

        PredictionResult {
            predicted_original_sizes: predictions,
            confidence_scores: confidences,
            exact_matches: 0, // Will be set during validation
            total_predictions: samples_to_predict.len(),
            processing_time_ms: processing_time,
            average_confidence,
        }
    }

    /// Validate predictions against ground truth and return detailed metrics
    pub fn validate_predictions(
        &self,
        predictions: &PredictionResult,
        ground_truth: &[u32],
    ) -> Result<ValidationMetrics, PredictionError> {
        if predictions.predicted_original_sizes.len() != ground_truth.len() {
            return Err(PredictionError::DataLengthMismatch {
                encrypted_len: predictions.predicted_original_sizes.len(),
                original_len: ground_truth.len(),
            });
        }

        let mut exact_matches = 0;
        let mut within_1_byte = 0;
        let mut within_2_bytes = 0;
        let mut within_5_bytes = 0;
        let mut total_error = 0i64;
        let mut max_error = 0u32;

        for (predicted, &actual) in predictions
            .predicted_original_sizes
            .iter()
            .zip(ground_truth.iter())
        {
            let error = (*predicted as i32 - actual as i32).abs() as u32;
            total_error += error as i64;
            max_error = max_error.max(error);

            if error == 0 {
                exact_matches += 1;
            }
            if error <= 1 {
                within_1_byte += 1;
            }
            if error <= 2 {
                within_2_bytes += 1;
            }
            if error <= 5 {
                within_5_bytes += 1;
            }
        }

        let metrics = ValidationMetrics {
            exact_matches,
            within_1_byte,
            within_2_bytes,
            within_5_bytes,
            total_samples: ground_truth.len(),
            average_error: total_error as f64 / ground_truth.len() as f64,
            max_error,
        };

        // Print validation results
        println!("Validation Results:");
        println!(
            "  Exact matches: {}/{} ({:.1}%)",
            metrics.exact_matches,
            metrics.total_samples,
            100.0 * metrics.exact_matches as f64 / metrics.total_samples as f64
        );
        println!(
            "  Within ±1 byte: {}/{} ({:.1}%)",
            metrics.within_1_byte,
            metrics.total_samples,
            100.0 * metrics.within_1_byte as f64 / metrics.total_samples as f64
        );
        println!(
            "  Within ±2 bytes: {}/{} ({:.1}%)",
            metrics.within_2_bytes,
            metrics.total_samples,
            100.0 * metrics.within_2_bytes as f64 / metrics.total_samples as f64
        );
        println!(
            "  Within ±5 bytes: {}/{} ({:.1}%)",
            metrics.within_5_bytes,
            metrics.total_samples,
            100.0 * metrics.within_5_bytes as f64 / metrics.total_samples as f64
        );
        println!("  Average error: {:.2} bytes", metrics.average_error);
        println!("  Maximum error: {} bytes", metrics.max_error);

        Ok(metrics)
    }

    /// Get recommended free atom size for MP4 moov reservation
    pub fn recommend_free_size(&self, profile: &AudioProfile) -> u32 {
        let base_estimate = profile.estimated_moov_size();

        // Apply safety margin based on training data quality
        let safety_multiplier = if self.is_trained && self.training_samples > 100 {
            1.15 // 15% buffer for well-trained predictor
        } else if self.is_trained {
            1.25 // 25% buffer for limited training data
        } else {
            1.5 // 50% buffer for untrained predictor
        };

        (base_estimate as f64 * safety_multiplier).ceil() as u32
    }

    // Private implementation methods

    fn validate_sample_pair(&self, encrypted: u32, original: u32) -> Result<(), PredictionError> {
        // Check 16-byte alignment
        if encrypted % 16 != 0 {
            return Err(PredictionError::NotAligned { size: encrypted });
        }

        // Validate padding (0-15 bytes for alignment, not 1-16 for PKCS#7)
        let padding = encrypted - original;
        if padding >= 16 {
            return Err(PredictionError::InvalidPadding {
                padding,
                encrypted,
                original,
            });
        }

        // Validate AAC frame size range
        if original < 100 || original > 4000 {
            return Err(PredictionError::InvalidAacSize { size: original });
        }

        Ok(())
    }

    fn build_transition_patterns(&mut self, encrypted_sizes: &[u32], original_sizes: &[u32]) {
        for window in encrypted_sizes.windows(2).zip(original_sizes.windows(2)) {
            let (enc_pair, orig_pair) = window;
            let enc_delta = enc_pair[1] as i32 - enc_pair[0] as i32;
            let orig_delta = orig_pair[1] as i32 - orig_pair[0] as i32;

            self.transition_patterns
                .entry(enc_delta)
                .or_insert_with(Vec::new)
                .push(orig_delta);
        }
    }

    fn calculate_confidence_scores(&mut self) {
        for (encrypted_size, originals) in &self.size_patterns {
            let unique_count = {
                let mut sorted = originals.clone();
                sorted.sort_unstable();
                sorted.dedup();
                sorted.len()
            };

            // Higher confidence for fewer unique mappings and more samples
            let base_confidence = 1.0 / unique_count as f64;
            let sample_boost = (originals.len() as f64).ln() / 10.0; // Logarithmic boost
            let confidence = (base_confidence + sample_boost).min(1.0);

            self.pattern_confidence.insert(*encrypted_size, confidence);
        }
    }

    fn analyze_bitrate_patterns(&mut self, encrypted_sizes: &[u32], original_sizes: &[u32]) {
        for (&encrypted, &original) in encrypted_sizes.iter().zip(original_sizes.iter()) {
            let padding = encrypted - original;

            let profile =
                self.bitrate_profiles
                    .entry(encrypted)
                    .or_insert_with(|| BitrateProfile {
                        avg_original_size: 0.0,
                        size_variance: 0.0,
                        common_padding_values: Vec::new(),
                        sample_count: 0,
                    });

            // Update running statistics
            let n = profile.sample_count as f64;
            profile.avg_original_size =
                (profile.avg_original_size * n + original as f64) / (n + 1.0);
            profile.common_padding_values.push(padding);
            profile.sample_count += 1;
        }

        // Calculate variance and sort padding values
        for profile in self.bitrate_profiles.values_mut() {
            profile.common_padding_values.sort_unstable();

            if profile.sample_count > 1 {
                let mean = profile.avg_original_size;
                let variance_sum: f64 = profile
                    .common_padding_values
                    .iter()
                    .map(|&padding| {
                        let original = profile.avg_original_size; // Approximation
                        let diff = original - mean;
                        diff * diff
                    })
                    .sum();
                profile.size_variance = variance_sum / (profile.sample_count - 1) as f64;
            }
        }
    }

    fn build_frame_size_distribution(&mut self, original_sizes: &[u32]) {
        let mut sizes = original_sizes.to_vec();
        sizes.sort_unstable();
        sizes.dedup();
        self.frame_size_distribution = sizes;
    }

    fn predict_single_sample(
        &self,
        encrypted_size: u32,
        all_encrypted: &[u32],
        current_index: usize,
        previous_predictions: &[u32],
    ) -> (u32, f64) {
        // Strategy 1: Direct pattern lookup
        if let Some(candidates) = self.size_patterns.get(&encrypted_size) {
            if candidates.len() == 1 {
                return (candidates[0], 1.0);
            }

            // Strategy 2: Use sequential context
            if current_index > 0 && !previous_predictions.is_empty() {
                if let Some(best_candidate) = self.select_best_candidate_with_context(
                    candidates,
                    encrypted_size,
                    all_encrypted[current_index - 1],
                    previous_predictions[previous_predictions.len() - 1],
                ) {
                    let confidence = self
                        .pattern_confidence
                        .get(&encrypted_size)
                        .cloned()
                        .unwrap_or(0.6);
                    return (best_candidate, confidence);
                }
            }

            // Strategy 3: Most common size for this encrypted size
            let most_common = Self::most_frequent(candidates);
            let confidence = self
                .pattern_confidence
                .get(&encrypted_size)
                .cloned()
                .unwrap_or(0.4);

            (most_common, confidence)
        } else {
            // Strategy 4: Predict from 16-byte alignment with AAC knowledge
            let predicted = self.predict_from_alignment(encrypted_size);
            (predicted, 0.7) // Higher confidence due to alignment constraints
        }
    }

    fn predict_from_alignment(&self, encrypted_size: u32) -> u32 {
        // With 16-byte alignment, original size is in [encrypted-15, encrypted]
        let possible_originals: Vec<u32> = (0..16)
            .map(|padding| encrypted_size - padding)
            .filter(|&size| self.is_plausible_aac_size(size))
            .collect();

        match possible_originals.len() {
            0 => encrypted_size.saturating_sub(8), // Fallback to middle padding
            1 => possible_originals[0],
            _ => self.select_most_likely_original(&possible_originals, encrypted_size),
        }
    }

    fn select_most_likely_original(&self, candidates: &[u32], encrypted_size: u32) -> u32 {
        candidates
            .iter()
            .max_by_key(|&&original| {
                let padding = encrypted_size - original;
                let mut score = 0;

                // Prefer smaller padding (more common in practice)
                if padding <= 8 {
                    score += 10;
                }
                if padding <= 4 {
                    score += 5;
                }

                // AAC frames often align to 4-byte boundaries
                if original % 4 == 0 {
                    score += 3;
                }

                // Prefer sizes we've seen before
                if self
                    .frame_size_distribution
                    .binary_search(&original)
                    .is_ok()
                {
                    score += 15;
                }

                // Prefer typical AAC frame sizes
                if self.is_typical_aac_frame_size(original) {
                    score += 8;
                }

                score
            })
            .copied()
            .unwrap_or(candidates[0])
    }

    fn select_best_candidate_with_context(
        &self,
        candidates: &[u32],
        current_encrypted: u32,
        prev_encrypted: u32,
        prev_original: u32,
    ) -> Option<u32> {
        let encrypted_delta = current_encrypted as i32 - prev_encrypted as i32;

        if let Some(orig_deltas) = self.transition_patterns.get(&encrypted_delta) {
            let expected_orig_delta = Self::most_frequent_i32(orig_deltas);
            let expected_original = (prev_original as i32 + expected_orig_delta) as u32;

            candidates
                .iter()
                .min_by_key(|&&candidate| (candidate as i32 - expected_original as i32).abs())
                .copied()
        } else {
            None
        }
    }

    fn is_plausible_aac_size(&self, size: u32) -> bool {
        size >= 100 && size <= 4000 && size > 0
    }

    fn is_typical_aac_frame_size(&self, size: u32) -> bool {
        // Common AAC VBR frame size ranges
        (size >= 200 && size <= 800) || // Typical music
        (size >= 100 && size <= 300) || // Low bitrate/simple content
        (size >= 800 && size <= 1500) // High bitrate/complex content
    }

    fn most_frequent(values: &[u32]) -> u32 {
        let mut counts = HashMap::new();
        for &value in values {
            *counts.entry(value).or_insert(0) += 1;
        }

        *counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(value, _)| value)
            .unwrap_or(&values[0])
    }

    fn most_frequent_i32(values: &[i32]) -> i32 {
        let mut counts = HashMap::new();
        for &value in values {
            *counts.entry(value).or_insert(0) += 1;
        }

        *counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(value, _)| value)
            .unwrap_or(&values[0])
    }
}

impl PredictionResult {
    fn empty() -> Self {
        Self {
            predicted_original_sizes: Vec::new(),
            confidence_scores: Vec::new(),
            exact_matches: 0,
            total_predictions: 0,
            processing_time_ms: 0.0,
            average_confidence: 0.0,
        }
    }

    /// Calculate size reduction estimate
    pub fn calculate_size_reduction(&self, encrypted_sizes: &[u32]) -> (u64, f64) {
        let total_encrypted: u64 = encrypted_sizes.iter().map(|&x| x as u64).sum();
        let total_predicted: u64 = self
            .predicted_original_sizes
            .iter()
            .map(|&x| x as u64)
            .sum();

        let reduction = total_encrypted.saturating_sub(total_predicted);
        let percentage = if total_encrypted > 0 {
            100.0 * reduction as f64 / total_encrypted as f64
        } else {
            0.0
        };

        (reduction, percentage)
    }
}

impl ValidationMetrics {
    /// Get accuracy percentage for exact matches
    pub fn exact_accuracy(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            100.0 * self.exact_matches as f64 / self.total_samples as f64
        }
    }

    /// Get accuracy percentage within 1 byte
    pub fn near_accuracy(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            100.0 * self.within_1_byte as f64 / self.total_samples as f64
        }
    }

    /// Check if metrics meet quality thresholds
    pub fn meets_quality_threshold(&self) -> bool {
        self.exact_accuracy() >= 80.0 && self.near_accuracy() >= 95.0
    }
}

/// Generate realistic test data for AAC VBR with 16-byte alignment
pub fn generate_test_data(num_samples: usize) -> (Vec<u32>, Vec<u32>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut encrypted_sizes = Vec::with_capacity(num_samples);
    let mut original_sizes = Vec::with_capacity(num_samples);

    let mut complexity_state = 0.5; // Audio complexity evolution

    for i in 0..num_samples {
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let rand = (hasher.finish() % 1000) as f64 / 1000.0;

        // Evolve complexity gradually (realistic for music)
        complexity_state += (rand - 0.5) * 0.1;
        complexity_state = complexity_state.clamp(0.0, 1.0);

        // AAC frame size based on complexity (VBR behavior)
        let base_size = 200 + (complexity_state * 600.0) as u32;
        let noise = ((rand - 0.5) * 100.0) as i32;
        let original_size = (base_size as i32 + noise).clamp(150, 1500) as u32;

        // Apply 16-byte alignment
        let encrypted_size = (original_size + 15) & 0xFFFFFFF0;

        original_sizes.push(original_size);
        encrypted_sizes.push(encrypted_size);
    }

    (encrypted_sizes, original_sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_validation() {
        let mut predictor = AacSamplePredictor::new();

        let original = vec![357, 344, 329];
        let encrypted: Vec<u32> = original
            .iter()
            .map(|&size| (size + 15) & 0xFFFFFFF0)
            .collect();

        assert!(predictor.train(&encrypted, &original).is_ok());
    }

    #[test]
    fn test_invalid_alignment() {
        let mut predictor = AacSamplePredictor::new();

        let encrypted = vec![367]; // Not 16-byte aligned
        let original = vec![350];

        assert!(predictor.train(&encrypted, &original).is_err());
    }

    #[test]
    fn test_prediction_accuracy() {
        let (encrypted, original) = generate_test_data(1000);
        let mut predictor = AacSamplePredictor::new();

        predictor
            .train(&encrypted[..200], &original[..200])
            .unwrap();
        let results = predictor.predict(&encrypted, 200);

        let metrics = predictor
            .validate_predictions(&results, &original[200..])
            .unwrap();

        // With alignment, should achieve high accuracy
        assert!(
            metrics.exact_accuracy() > 70.0,
            "Expected >70% exact accuracy, got {:.1}%",
            metrics.exact_accuracy()
        );
        assert!(
            metrics.near_accuracy() > 90.0,
            "Expected >90% near accuracy, got {:.1}%",
            metrics.near_accuracy()
        );
    }

    #[test]
    fn test_audio_profile() {
        let profile = AudioProfile::new(44100, 2, 180000); // 3 minutes
        let sample_count = profile.estimated_sample_count();
        let moov_size = profile.estimated_moov_size();

        assert!(sample_count > 7000); // ~7826 for 3 minutes
        assert!(moov_size > 90000); // Should be reasonable size
    }

    #[test]
    fn test_empty_data_handling() {
        let mut predictor = AacSamplePredictor::new();

        let result = predictor.train(&[], &[]);
        assert!(result.is_err());

        let results = predictor.predict(&[], 0);
        assert_eq!(results.total_predictions, 0);
    }
}
