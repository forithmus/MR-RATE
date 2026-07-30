from mrrate_report_training.targets import make_report_target


def test_complete_report_preserves_statement_order():
    findings = (
        "There is no acute intracranial abnormality.\n"
        "A small chronic infarct is present.\n"
        "Cannot exclude a tiny focus of hemorrhage."
    )
    target = make_report_target("study", findings)
    assert target.findings == findings
    assert target.text == findings


def test_inline_whitespace_is_normalized_but_lines_are_preserved():
    target = make_report_target("study", "  First   line. \n\n Second line. ")
    assert target.text == "First line.\nSecond line."
