# Task 1: Verified recursive self-training
Loop: pseudo-label the pool -> verify (teacher confidence >= 0.9 and agreement with an independent char-n-gram logistic regression fitted on the seed set) -> retrain a copy on seed + accepted examples -> accept only if dev macro-F1 does not drop, else roll back -> stop after 2 consecutive rejections (max 5 rounds).
Variants compared: `verified`, `confidence_only`, `naive` (no rollback). Logged per round: candidates, pseudo-label precision (vs hidden gold), dev F1/ECE, class entropy, decision.
Run: `python -m task1_recursive.run`. The verifier is a proxy, so this is iterative self-training, not recursive self-improvement.
