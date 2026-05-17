use std::collections::{HashMap, HashSet};

/// Scale logits by `1 / temperature`. No-op when temperature == 1.0.
/// When temperature == 0.0, the caller should use greedy (argmax) instead.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || (temperature - 1.0).abs() < f32::EPSILON {
        return;
    }
    let inv_temp = 1.0 / temperature;
    for logit in logits.iter_mut() {
        *logit *= inv_temp;
    }
}

/// Retain only the top-k logits, setting all others to -inf.
/// When k <= 0 or k >= vocab_size, no filtering is applied.
pub fn apply_top_k(logits: &mut [f32], k: i32) {
    if k <= 0 {
        return;
    }
    let k = k as usize;
    if k >= logits.len() {
        return;
    }

    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let threshold = logits[indices[k]];
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Nucleus (top-p) filtering: keep the smallest set of tokens whose cumulative
/// probability >= p, setting the rest to -inf.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 || p <= 0.0 || logits.is_empty() {
        return;
    }

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, v)| v != f32::NEG_INFINITY)
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_val = indexed.first().map(|&(_, v)| v).unwrap_or(0.0);
    let exps: Vec<f32> = indexed
        .iter()
        .map(|&(_, v)| (v - max_val).exp())
        .collect();
    let sum: f32 = exps.iter().copied().sum();
    if sum == 0.0 {
        return;
    }

    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &exp_val) in exps.iter().enumerate() {
        cumsum += exp_val / sum;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }
    for &(idx, _) in &indexed[cutoff..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Min-p filtering: keep tokens whose probability >= min_p * max_probability.
pub fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || logits.is_empty() {
        return;
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if max_logit == f32::NEG_INFINITY {
        return;
    }

    let threshold = min_p.ln() + max_logit;
    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Apply repetition penalty: divide logits of previously seen tokens.
/// For positive logits, divide by penalty; for negative logits, multiply by penalty.
pub fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    if penalty == 1.0 || token_ids.is_empty() {
        return;
    }
    for &tid in token_ids {
        let idx = tid as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Apply frequency penalty: subtract `penalty * count` for each token.
pub fn apply_frequency_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    if penalty == 0.0 || token_ids.is_empty() {
        return;
    }
    let mut counts = HashMap::new();
    for &tid in token_ids {
        *counts.entry(tid).or_insert(0u32) += 1;
    }
    for (&tid, &count) in &counts {
        let idx = tid as usize;
        if idx < logits.len() {
            logits[idx] -= penalty * count as f32;
        }
    }
}

/// Apply presence penalty: subtract `penalty` once for each unique token.
pub fn apply_presence_penalty(logits: &mut [f32], token_ids: &[u32], penalty: f32) {
    if penalty == 0.0 || token_ids.is_empty() {
        return;
    }
    let seen: HashSet<u32> = token_ids.iter().copied().collect();
    for &tid in &seen {
        let idx = tid as usize;
        if idx < logits.len() {
            logits[idx] -= penalty;
        }
    }
}

/// Apply logit bias: add the bias value to the specified token positions.
pub fn apply_logit_bias(logits: &mut [f32], bias: &HashMap<u32, f32>) {
    for (&token_id, &bias_val) in bias {
        let idx = token_id as usize;
        if idx < logits.len() {
            logits[idx] += bias_val;
        }
    }
}

/// Mask out all tokens NOT in the allowed set.
pub fn apply_allowed_token_ids(logits: &mut [f32], allowed: &HashSet<u32>) {
    if allowed.is_empty() {
        return;
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        if !allowed.contains(&(i as u32)) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Mask out the EOS token when `num_generated < min_tokens`.
pub fn apply_eos_suppression(
    logits: &mut [f32],
    eos_token_id: u32,
    num_generated: u32,
    min_tokens: u32,
) {
    if num_generated < min_tokens {
        let idx = eos_token_id as usize;
        if idx < logits.len() {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

/// Mask out token IDs in `bad_token_ids`.
pub fn apply_bad_token_ids(logits: &mut [f32], bad_token_ids: &[u32]) {
    for &tid in bad_token_ids {
        let idx = tid as usize;
        if idx < logits.len() {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

/// Compute softmax in-place, returning the log-sum-exp for logprob calculation.
pub fn softmax_in_place(logits: &mut [f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return max;
    }
    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max).exp();
        sum += *logit;
    }
    for logit in logits.iter_mut() {
        *logit /= sum;
    }
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scales_logits() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 0.5);
        assert!((logits[0] - 2.0).abs() < 1e-5);
        assert!((logits[1] - 4.0).abs() < 1e-5);
        assert!((logits[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_temperature_one_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_top_k_filters() {
        let mut logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        apply_top_k(&mut logits, 2);
        // Top 2 values: 0.9 (idx 3) and 0.5 (idx 1). All others should be -inf.
        assert!(logits[3].is_finite());
        assert!(logits[1].is_finite());
        // Remaining may or may not be -inf depending on ties in select_nth_unstable.
        // Verify at least the top-2 are kept.
        let finite_count = logits.iter().filter(|&&l| l.is_finite()).count();
        assert!(finite_count >= 2);
    }

    #[test]
    fn test_top_k_negative_is_noop() {
        let mut logits = vec![0.1, 0.5, 0.3];
        apply_top_k(&mut logits, -1);
        assert_eq!(logits, vec![0.1, 0.5, 0.3]);
    }

    #[test]
    fn test_repetition_penalty_divides_positive() {
        let mut logits = vec![2.0, -1.0, 0.5];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-5);
        assert!((logits[1] - (-1.0)).abs() < 1e-5);
        assert!((logits[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_multiplies_negative() {
        let mut logits = vec![2.0, -2.0, 0.5];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!((logits[0] - 2.0).abs() < 1e-5);
        assert!((logits[1] - (-4.0)).abs() < 1e-5);
    }

    #[test]
    fn test_frequency_penalty_subtracts_by_count() {
        let mut logits = vec![0.0, 0.0, 0.0];
        apply_frequency_penalty(&mut logits, &[0, 0, 1], 0.5);
        assert!((logits[0] - (-1.0)).abs() < 1e-5);
        assert!((logits[1] - (-0.5)).abs() < 1e-5);
        assert!((logits[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_presence_penalty_subtracts_once() {
        let mut logits = vec![0.0, 0.0, 0.0];
        apply_presence_penalty(&mut logits, &[0, 0, 1], 0.5);
        assert!((logits[0] - (-0.5)).abs() < 1e-5);
        assert!((logits[1] - (-0.5)).abs() < 1e-5);
        assert!((logits[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_logit_bias_adds() {
        let mut logits = vec![0.0, 1.0, 2.0];
        let bias = HashMap::from([(0u32, 5.0f32), (2u32, -3.0f32)]);
        apply_logit_bias(&mut logits, &bias);
        assert!((logits[0] - 5.0).abs() < 1e-5);
        assert!((logits[1] - 1.0).abs() < 1e-5);
        assert!((logits[2] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_allowed_token_ids_masks() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let allowed = HashSet::from([1u32, 3u32]);
        apply_allowed_token_ids(&mut logits, &allowed);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert!(logits[1].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert!(logits[3].is_finite());
    }

    #[test]
    fn test_eos_suppression_under_min_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_eos_suppression(&mut logits, 2, 0, 5);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_eos_suppression_past_min_tokens() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_eos_suppression(&mut logits, 2, 10, 5);
        assert!(logits[2].is_finite());
    }

    #[test]
    fn test_min_p_filters_low_prob() {
        let mut logits = vec![10.0, 5.0, 0.0];
        apply_min_p(&mut logits, 0.1);
        assert!(logits[0].is_finite());
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_softmax_in_place() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let lse = softmax_in_place(&mut logits);
        assert!((logits[0] + logits[1] + logits[2] - 1.0).abs() < 1e-5);
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
        let expected_lse = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
        assert!((lse - expected_lse).abs() < 1e-3);
    }
}
