import pytest
from src.brain_image.utils import update_config_with_nested_key


def test_update_flat_key():
    config = {"a": 1, "b": 2}
    updated = update_config_with_nested_key("a", 42, config)
    assert updated["a"] == 42
    assert updated["b"] == 2
    # Original config should not be mutated
    assert config["a"] == 1


def test_update_nested_key():
    config = {"a": {"b": {"c": 1}}, "x": 2}
    updated = update_config_with_nested_key("a.b.c", 99, config)
    assert updated["a"]["b"]["c"] == 99
    assert updated["x"] == 2
    # Original config should not be mutated
    assert config["a"]["b"]["c"] == 1


def test_update_new_nested_key():
    config = {"a": {"b": {}}}
    updated = update_config_with_nested_key("a.b.c", 123, config)
    assert updated["a"]["b"]["c"] == 123


def test_create_intermediate_nested_keys():
    config = {"a": {}}
    updated = update_config_with_nested_key("a.b.c", 5, config)
    assert updated["a"]["b"]["c"] == 5
    # Original config should not be mutated
    assert "b" not in config["a"]


def test_create_full_nested_path_from_empty():
    config = {}
    updated = update_config_with_nested_key("a.b.c", 42, config)
    assert updated["a"]["b"]["c"] == 42
    assert config == {}


def test_create_deep_nested_keys():
    config = {"a": {}}
    updated = update_config_with_nested_key("a.b.c.d", 99, config)
    assert updated["a"]["b"]["c"]["d"] == 99
    assert config == {"a": {}}


def test_update_nonexistent_path():
    config = {"a": {}}
    # Should create intermediate keys if not present
    updated = update_config_with_nested_key("a.b.c", 5, config)
    assert updated["a"]["b"]["c"] == 5
    assert config == {"a": {}}


def test_update_top_level_new_key():
    config = {"a": 1}
    updated = update_config_with_nested_key("z", 7, config)
    assert updated["z"] == 7
    assert updated["a"] == 1
