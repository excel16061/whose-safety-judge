from wsj import baselines as B


def test_keyword_rule_whole_words_case_insensitive():
    out = B.keyword_rule(["That is STUPID.", "stupidity is not listed", "a gun", "nothing here"])
    assert out.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_constant_and_majority():
    assert B.constant_baseline(3, 0.37).tolist() == [0.37] * 3
    assert B.majority_baseline(2, 0.6).tolist() == [1.0, 1.0]
    assert B.majority_baseline(2, 0.4).tolist() == [0.0, 0.0]
