"""Tests for retrieval tool wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.tools import retrieve_bundle


def test_retrieve_bundle_passes_allowed_ids_for_known_category():
    hybrid = MagicMock()
    hybrid.get_ids_for_category.return_value = {1, 2, 3}
    hybrid.retrieve_context.return_value = ([], [])

    retrieve_bundle(hybrid, "quality issues", top_k=5, category_filter="All_Beauty")

    hybrid.get_ids_for_category.assert_called_once_with("All_Beauty")
    hybrid.retrieve_context.assert_called_once()
    _, kwargs = hybrid.retrieve_context.call_args
    assert kwargs["allowed_ids"] == {1, 2, 3}
    assert kwargs["top_k"] == 5


def test_retrieve_bundle_skips_filter_for_unknown_category_name():
    hybrid = MagicMock()
    hybrid.retrieve_context.return_value = ([], [])

    retrieve_bundle(hybrid, "q", category_filter="NotInCorpus")

    hybrid.get_ids_for_category.assert_not_called()
    _, kwargs = hybrid.retrieve_context.call_args
    assert kwargs["allowed_ids"] is None
