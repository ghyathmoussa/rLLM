use std::collections::HashMap;

/// Draft tokens proposed ahead of the target model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftProposal {
    pub token_ids: Vec<u32>,
}

impl DraftProposal {
    pub fn empty() -> Self {
        Self { token_ids: Vec::new() }
    }
}

/// Common interface for speculative decoding proposers.
pub trait SpeculativeProposer {
    fn propose(&self, context_token_ids: &[u32], max_tokens: usize) -> DraftProposal;
}

/// N-gram proposer that repeats the continuation after the longest suffix match.
#[derive(Debug, Clone)]
pub struct NGramProposer {
    min_ngram: usize,
    max_ngram: usize,
}

impl NGramProposer {
    pub fn new(min_ngram: usize, max_ngram: usize) -> Self {
        assert!(min_ngram > 0);
        assert!(max_ngram >= min_ngram);
        Self { min_ngram, max_ngram }
    }
}

impl Default for NGramProposer {
    fn default() -> Self {
        Self::new(2, 4)
    }
}

impl SpeculativeProposer for NGramProposer {
    fn propose(&self, context_token_ids: &[u32], max_tokens: usize) -> DraftProposal {
        if max_tokens == 0 || context_token_ids.len() <= self.min_ngram {
            return DraftProposal::empty();
        }

        let max_n = self.max_ngram.min(context_token_ids.len() - 1);
        for n in (self.min_ngram..=max_n).rev() {
            let suffix = &context_token_ids[context_token_ids.len() - n..];
            for start in (0..context_token_ids.len() - n).rev() {
                if &context_token_ids[start..start + n] == suffix {
                    let continuation_start = start + n;
                    let continuation_end =
                        (continuation_start + max_tokens).min(context_token_ids.len());
                    if continuation_start < continuation_end {
                        return DraftProposal {
                            token_ids: context_token_ids[continuation_start..continuation_end]
                                .to_vec(),
                        };
                    }
                }
            }
        }
        DraftProposal::empty()
    }
}

/// Placeholder for a small draft model proposer.
#[derive(Debug, Clone)]
pub struct DraftModelProposer {
    pub model_id: String,
    pub max_draft_tokens: usize,
}

impl SpeculativeProposer for DraftModelProposer {
    fn propose(&self, _context_token_ids: &[u32], _max_tokens: usize) -> DraftProposal {
        DraftProposal::empty()
    }
}

/// Placeholder for future EAGLE-style feature extrapolation.
#[derive(Debug, Clone, Default)]
pub struct EagleProposer;

impl SpeculativeProposer for EagleProposer {
    fn propose(&self, _context_token_ids: &[u32], _max_tokens: usize) -> DraftProposal {
        DraftProposal::empty()
    }
}

/// Verify draft tokens against target-model accepted tokens.
pub fn accept_matching_prefix(draft: &[u32], target: &[u32]) -> Vec<u32> {
    draft.iter().zip(target).take_while(|(a, b)| a == b).map(|(token, _)| *token).collect()
}

/// Per-request speculative state.
#[derive(Debug, Clone, Default)]
pub struct SpeculativeState {
    drafts: HashMap<u64, Vec<u32>>,
}

impl SpeculativeState {
    pub fn set_draft(&mut self, request_key: u64, token_ids: Vec<u32>) {
        self.drafts.insert(request_key, token_ids);
    }

    pub fn take_draft(&mut self, request_key: u64) -> Option<Vec<u32>> {
        self.drafts.remove(&request_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_proposer_reuses_prior_continuation() {
        let proposer = NGramProposer::new(2, 3);
        let proposal = proposer.propose(&[1, 2, 3, 4, 2, 3], 2);
        assert_eq!(proposal.token_ids, vec![4, 2]);
    }

    #[test]
    fn accepts_only_matching_prefix() {
        assert_eq!(accept_matching_prefix(&[1, 2, 3], &[1, 2, 9]), vec![1, 2]);
    }
}
