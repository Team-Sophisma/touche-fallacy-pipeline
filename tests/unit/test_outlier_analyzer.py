import pandas as pd

from src.infrastructure.eda.outlier_analyzer import OutlierAnalyzer


def test_outlier_analyzer_reports_and_flags_candidates_only():
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "label": ["x", "x", "x", "x", "x"],
        "fallacy_type": ["x", "x", "x", "x", "x"],
        "argument_word_count": [10, 11, 12, 13, 100],
    })

    analyzer = OutlierAnalyzer(
        metrics=["argument_word_count"],
        iqr_multiplier=1.5,
    )

    report = analyzer.analyze(df)
    candidates = analyzer.flag_dataframe(df)

    assert report["method"] == "iqr"
    assert report["unique_outlier_ids"] == ["e"]
    assert report["global"][0]["outlier_count"] == 1
    assert list(candidates["id"]) == ["e"]
    assert "is_outlier" not in df.columns
