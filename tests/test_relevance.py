"""Quick unit tests for relevance module."""
import json
from finacumen.fm.relevance import _parse_json_array, _format_entries, RELEVANCE_THRESHOLD


def test_parse_json_array_answer_tag():
    result = _parse_json_array('<Answer> [{"i": 0, "score": 0.7}] </Answer>')
    assert result == [{"i": 0, "score": 0.7}], f"got {result}"


def test_parse_json_array_raw():
    result = _parse_json_array('[{"i": 0, "score": 0.5}]')
    assert result == [{"i": 0, "score": 0.5}], f"got {result}"


def test_parse_json_array_invalid():
    assert _parse_json_array("no json here") is None
    assert _parse_json_array("<Answer>not json</Answer>") is None


def test_parse_json_array_markdown_fence():
    text = '<Answer>\n```json\n[{"i": 0, "score": 0.9}]\n```\n</Answer>'
    result = _parse_json_array(text)
    assert result == [{"i": 0, "score": 0.9}], f"got {result}"


def test_format_entries():
    entries = [
        {
            "question": "Test question 1",
            "experience": {"findings": ["f1", "f2"], "cautions": ["c1"]},
        },
        {
            "question": "Test question 2",
            "experience": "raw experience text",
        },
    ]
    output = _format_entries(entries)
    assert "Entry_0" in output
    assert "Entry_1" in output
    assert "Test question 1" in output
    assert "f1" in output
    assert "c1" in output
    assert "raw experience text" in output


def test_threshold():
    assert RELEVANCE_THRESHOLD == 0.5


if __name__ == "__main__":
    test_parse_json_array_answer_tag()
    test_parse_json_array_raw()
    test_parse_json_array_invalid()
    test_parse_json_array_markdown_fence()
    test_format_entries()
    test_threshold()
    print("All tests passed")
