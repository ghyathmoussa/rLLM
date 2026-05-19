/// Compute the log probability of a specific token from raw logits.
///
/// Returns `(logprob_of_token, max_logit)` where `logprob_of_token` is in
/// natural log space.
pub fn compute_logprob(logits: &[f32], token_id: u32) -> (f32, f32) {
    if logits.is_empty() {
        return (0.0, 0.0);
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return (f32::NEG_INFINITY, max);
    }

    let idx = token_id as usize;
    let logit_val = if idx < logits.len() {
        logits[idx]
    } else {
        return (f32::NEG_INFINITY, max);
    };

    // log-softmax: logit - log(sum(exp(logits)))
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();
    let logprob = logit_val - log_sum_exp;

    (logprob, max)
}

/// Extract the top-N log probabilities from raw logits.
///
/// Returns a sorted vector of `(token_id, logprob)` pairs, descending by logprob.
/// Returns fewer than `n` entries if the vocab is smaller than `n`.
pub fn compute_top_n_logprobs(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() || n == 0 {
        return Vec::new();
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return Vec::new();
    }

    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v != f32::NEG_INFINITY)
        .map(|(i, &logit)| (i, logit - log_sum_exp))
        .collect();

    // Partial sort: we only need top-n.
    let limit = n.min(indexed.len());
    if limit < indexed.len() {
        indexed.select_nth_unstable_by(limit, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(limit);
    }
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed.into_iter().map(|(i, lp)| (i as u32, lp)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_logprob_highest_token() {
        let logits = vec![1.0, 2.0, 3.0];
        let (lp, _) = compute_logprob(&logits, 2);
        // Token 2 has highest logit, logprob = log(exp(3) / (exp(1)+exp(2)+exp(3)))
        // This is approximately -0.09, not exactly 0.
        assert!(lp > -0.5);
        assert!(lp <= 0.0);
    }

    #[test]
    fn test_compute_logprob_lowest_token() {
        let logits = vec![1.0, 2.0, 3.0];
        let (lp, _) = compute_logprob(&logits, 0);
        assert!(lp < -1.0);
    }

    #[test]
    fn test_compute_logprob_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let (lp0, _) = compute_logprob(&logits, 0);
        let (lp1, _) = compute_logprob(&logits, 1);
        let (lp2, _) = compute_logprob(&logits, 2);
        let sum = lp0.exp() + lp1.exp() + lp2.exp();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_top_n_logprobs_returns_sorted() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let top = compute_top_n_logprobs(&logits, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // highest
        assert_eq!(top[1].0, 2); // second highest
        assert!(top[0].1 > top[1].1);
    }

    #[test]
    fn test_top_n_logprobs_n_exceeds_vocab() {
        let logits = vec![1.0, 2.0];
        let top = compute_top_n_logprobs(&logits, 10);
        assert_eq!(top.len(), 2);
    }
}
