import math

import pytest

from mrrate_report_training.nlg_metrics import (
    compute_nlg_metrics,
    corpus_bleu,
    rouge_l,
    rouge_n,
    sentence_bleu,
    tokenize,
)


def test_tokenize_normalizes_case_and_punctuation():
    assert tokenize("There is NO acute infarct.") == [
        "there",
        "is",
        "no",
        "acute",
        "infarct",
    ]


def test_identical_texts_score_one():
    text = "A chronic lacunar infarct is present in the left thalamus."
    scores = corpus_bleu([text], [text])
    for order in range(1, 5):
        assert scores[f"bleu{order}"] == pytest.approx(1.0)
    assert rouge_l(text, text) == pytest.approx(1.0)
    assert rouge_n(text, text, 2) == pytest.approx(1.0)


def test_disjoint_texts_score_zero():
    assert corpus_bleu(["alpha beta gamma"], ["delta epsilon zeta"])[
        "bleu1"
    ] == pytest.approx(0.0)
    assert rouge_l("alpha beta", "gamma delta") == pytest.approx(0.0)


def test_corpus_bleu_hand_computed():
    # hyp unigrams: the(2->clipped 1), cat(1) => 2/3; brevity: |hyp|=3 < |ref|=4
    scores = corpus_bleu(["the cat sat down"], ["the cat the"])
    brevity = math.exp(1.0 - 4.0 / 3.0)
    assert scores["bleu1"] == pytest.approx(brevity * 2.0 / 3.0)


def test_rouge_l_hand_computed():
    # LCS("the cat sat", "the sat cat") = 2 ("the cat" or "the sat")
    score = rouge_l("the cat sat", "the sat cat")
    precision = recall = 2.0 / 3.0
    assert score == pytest.approx(2 * precision * recall / (precision + recall))


def test_empty_hypothesis_is_zero_not_crash():
    assert sentence_bleu("some reference text", "") == 0.0
    assert rouge_l("some reference text", "") == 0.0
    scores = corpus_bleu(["a b"], [""])
    assert scores["bleu1"] == 0.0


def test_sentence_bleu_zero_for_disjoint_and_not_inflated_when_short():
    # No unigram overlap must score 0 despite smoothing.
    assert sentence_bleu("alpha beta gamma", "delta epsilon zeta") == 0.0
    # A one-token hypothesis is scored on the orders it supports, not
    # credited with perfect higher-order precision.
    reference = "the cat sat on the mat"
    assert sentence_bleu(reference, "the") < sentence_bleu(reference, "the cat sat")


def test_compute_nlg_metrics_shapes():
    references = ["no acute infarct", "there is hemorrhage"]
    hypotheses = ["no acute infarct", "there is no hemorrhage"]
    corpus, per_sample = compute_nlg_metrics(references, hypotheses)
    assert corpus["samples"] == 2
    assert len(per_sample) == 2
    assert per_sample[0]["rougeL_f1"] == pytest.approx(1.0)
    assert 0.0 < corpus["bleu1"] <= 1.0
    with pytest.raises(ValueError):
        compute_nlg_metrics(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        compute_nlg_metrics([], [])


def test_bleu_matches_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")
    references = [
        "there is a chronic infarct in the right frontal lobe",
        "no acute intracranial abnormality is present",
        "mild diffuse cerebral atrophy with chronic microangiopathy",
    ]
    hypotheses = [
        "there is a chronic infarct in the left frontal lobe",
        "no acute intracranial hemorrhage is present",
        "mild cerebral atrophy and chronic small vessel disease",
    ]
    ours = corpus_bleu(references, hypotheses)["bleu4"] * 100.0
    theirs = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize="none",
        smooth_method="none",
        force=True,
        lowercase=True,
    ).score
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_rouge_matches_rouge_score_package():
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    reference = "there is a chronic infarct in the right frontal lobe"
    hypothesis = "a chronic infarct is seen in the left frontal lobe"
    theirs = scorer.score(reference, hypothesis)
    assert rouge_n(reference, hypothesis, 1) == pytest.approx(
        theirs["rouge1"].fmeasure, abs=1e-6
    )
    assert rouge_n(reference, hypothesis, 2) == pytest.approx(
        theirs["rouge2"].fmeasure, abs=1e-6
    )
    assert rouge_l(reference, hypothesis) == pytest.approx(
        theirs["rougeL"].fmeasure, abs=1e-6
    )
