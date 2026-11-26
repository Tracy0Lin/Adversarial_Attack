"""VQAv2 answer matching (same rule as prepare_vqa.py)."""


def blip_style_correct(pred_ans, batch_answers, batch_weights):
    """pred must be in annotated answers and tie for max weight among matches."""
    answers = [a[0] if isinstance(a, (list, tuple)) else a for a in batch_answers]
    weights = [float(w[0].cpu().numpy()) if hasattr(w[0], "cpu") else float(w[0]) for w in batch_weights]
    if pred_ans not in answers:
        return False
    matched_weight = weights[answers.index(pred_ans)]
    return matched_weight == max(weights)
