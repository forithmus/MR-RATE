"""NLG metrics between generated and ground-truth findings.

BLEU-1..4 (corpus-level with brevity penalty, plus smoothed per-sample
scores), ROUGE-1/2 F1, and ROUGE-L F1 are implemented here without third
party dependencies so evaluation runs on offline compute nodes. METEOR is
computed through nltk when it is importable and its wordnet data is
available; otherwise it is reported as null rather than failing the run.

CLI:
    python -m mrrate_report_training.nlg_metrics \
        --generated-csv generated_val.csv --output-json nlg_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text or "").lower())


def ngram_counts(tokens: list[str], order: int) -> Counter:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def _clipped_matches(
    reference: list[str], hypothesis: list[str], order: int
) -> tuple[int, int]:
    total = max(len(hypothesis) - order + 1, 0)
    if not total:
        return 0, 0
    overlap = ngram_counts(hypothesis, order) & ngram_counts(reference, order)
    return sum(overlap.values()), total


def corpus_bleu(
    references: list[str], hypotheses: list[str], max_order: int = 4
) -> dict[str, float]:
    """Corpus BLEU with geometric mean and brevity penalty (BLEU-1..N)."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses differ in length")
    if not references:
        raise ValueError("cannot score an empty corpus")
    matches = [0] * max_order
    totals = [0] * max_order
    reference_length = hypothesis_length = 0
    for reference_text, hypothesis_text in zip(references, hypotheses):
        reference = tokenize(reference_text)
        hypothesis = tokenize(hypothesis_text)
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)
        for order in range(1, max_order + 1):
            matched, total = _clipped_matches(reference, hypothesis, order)
            matches[order - 1] += matched
            totals[order - 1] += total
    if hypothesis_length == 0:
        return {f"bleu{order}": 0.0 for order in range(1, max_order + 1)}
    brevity = (
        1.0
        if hypothesis_length >= reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    scores = {}
    log_precisions = []
    for order in range(1, max_order + 1):
        if totals[order - 1] == 0 or matches[order - 1] == 0:
            log_precisions.append(None)
        else:
            log_precisions.append(
                math.log(matches[order - 1] / totals[order - 1])
            )
        usable = log_precisions[:order]
        if any(value is None for value in usable):
            scores[f"bleu{order}"] = 0.0
        else:
            scores[f"bleu{order}"] = brevity * math.exp(
                sum(usable) / len(usable)
            )
    scores["brevity_penalty"] = brevity
    return scores


def sentence_bleu(reference: str, hypothesis: str, max_order: int = 4) -> float:
    """Per-sample BLEU: exact unigram precision, add-one smoothing above.

    A hypothesis with no unigram overlap scores 0. Orders longer than the
    hypothesis are excluded instead of being credited as perfect.
    """

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)
    if not hypothesis_tokens:
        return 0.0
    log_precisions = []
    for order in range(1, max_order + 1):
        matched, total = _clipped_matches(
            reference_tokens, hypothesis_tokens, order
        )
        if total == 0:
            continue
        if order == 1:
            if matched == 0:
                return 0.0
            log_precisions.append(math.log(matched / total))
        else:
            log_precisions.append(math.log((matched + 1) / (total + 1)))
    brevity = (
        1.0
        if len(hypothesis_tokens) >= len(reference_tokens)
        else math.exp(1.0 - len(reference_tokens) / len(hypothesis_tokens))
    )
    return brevity * math.exp(sum(log_precisions) / len(log_precisions))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_n(reference: str, hypothesis: str, order: int) -> float:
    """ROUGE-N F1 on token n-grams."""

    reference_counts = ngram_counts(tokenize(reference), order)
    hypothesis_counts = ngram_counts(tokenize(hypothesis), order)
    reference_total = sum(reference_counts.values())
    hypothesis_total = sum(hypothesis_counts.values())
    if not reference_total or not hypothesis_total:
        return 0.0
    overlap = sum((reference_counts & hypothesis_counts).values())
    return _f1(overlap / hypothesis_total, overlap / reference_total)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right):
            if left_token == right_token:
                current.append(previous[index] + 1)
            else:
                current.append(max(previous[index + 1], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 from the longest common token subsequence."""

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    return _f1(lcs / len(hypothesis_tokens), lcs / len(reference_tokens))


def _meteor_scorer():
    try:
        from nltk.translate.meteor_score import meteor_score

        # Non-identical tokens force the wordnet lookup, so a missing nltk
        # corpus is detected here instead of mid-corpus.
        meteor_score([["infarct"]], ["hemorrhage"])
    except Exception:
        return None
    return meteor_score


def compute_nlg_metrics(
    references: list[str], hypotheses: list[str]
) -> tuple[dict, list[dict]]:
    """Corpus metrics plus one per-sample metrics dict per pair."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses differ in length")
    if not references:
        raise ValueError("cannot score an empty corpus")
    corpus = corpus_bleu(references, hypotheses)
    meteor = _meteor_scorer()
    per_sample = []
    for reference, hypothesis in zip(references, hypotheses):
        row = {
            "bleu4": sentence_bleu(reference, hypothesis),
            "rouge1_f1": rouge_n(reference, hypothesis, 1),
            "rouge2_f1": rouge_n(reference, hypothesis, 2),
            "rougeL_f1": rouge_l(reference, hypothesis),
            "reference_tokens": len(tokenize(reference)),
            "hypothesis_tokens": len(tokenize(hypothesis)),
        }
        if meteor is not None:
            row["meteor"] = meteor(
                [tokenize(reference)], tokenize(hypothesis)
            )
        per_sample.append(row)
    count = len(per_sample)
    corpus.update(
        {
            "rouge1_f1": sum(row["rouge1_f1"] for row in per_sample) / count,
            "rouge2_f1": sum(row["rouge2_f1"] for row in per_sample) / count,
            "rougeL_f1": sum(row["rougeL_f1"] for row in per_sample) / count,
            "sentence_bleu4": sum(row["bleu4"] for row in per_sample) / count,
            "meteor": (
                sum(row["meteor"] for row in per_sample) / count
                if meteor is not None
                else None
            ),
            "samples": count,
        }
    )
    return corpus, per_sample


def load_generated_csv(paths: list[str | Path]) -> list[dict]:
    """Load and concatenate generation CSVs (study_uid, findings_gt/pred)."""

    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"study_uid", "findings_gt", "findings_pred"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for row in reader:
                subject_id = str(row["study_uid"]).strip()
                if not subject_id:
                    raise ValueError(f"{path}: empty study_uid")
                if subject_id in seen:
                    raise ValueError(f"{path}: duplicate study_uid {subject_id}")
                seen.add(subject_id)
                rows.append(
                    {
                        "study_uid": subject_id,
                        "findings_gt": str(row["findings_gt"] or ""),
                        "findings_pred": str(row["findings_pred"] or ""),
                    }
                )
    if not rows:
        raise ValueError("No generated reports found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-csv", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--per-sample-csv")
    args = parser.parse_args()
    rows = load_generated_csv(args.generated_csv)
    corpus, per_sample = compute_nlg_metrics(
        [row["findings_gt"] for row in rows],
        [row["findings_pred"] for row in rows],
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, indent=2) + "\n")
    if args.per_sample_csv:
        fields = ["study_uid", *per_sample[0].keys()]
        with Path(args.per_sample_csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row, sample in zip(rows, per_sample):
                writer.writerow({"study_uid": row["study_uid"], **sample})
    print(json.dumps(corpus, indent=2), flush=True)


if __name__ == "__main__":
    main()
