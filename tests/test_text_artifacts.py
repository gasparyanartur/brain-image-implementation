import json

import pytest

from scripts.data.validate_text_artifacts import load_caption_records


def test_load_caption_records_rejects_empty_caption(tmp_path):
    path = tmp_path / "captions.jsonl"
    path.write_text(json.dumps({"path": "image.jpg", "caption": "  "}) + "\n")

    with pytest.raises(ValueError, match="Missing path or caption"):
        load_caption_records(path)


def test_load_caption_records_rejects_invalid_json(tmp_path):
    path = tmp_path / "captions.jsonl"
    path.write_text("not json\n")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_caption_records(path)
