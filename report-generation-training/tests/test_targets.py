from mrrate_report_training.targets import make_report_target


def test_complete_report_preserves_statement_order():
    statements = [
        "There is no acute intracranial abnormality.",
        "A small chronic infarct is present.",
        "Cannot exclude a tiny focus of hemorrhage.",
    ]
    target = make_report_target("study", statements)
    assert target.statements == tuple(statements)
    assert target.text == " ".join(statements)


def test_empty_report_has_explicit_none():
    assert make_report_target("study", []).text == "<NONE>"

