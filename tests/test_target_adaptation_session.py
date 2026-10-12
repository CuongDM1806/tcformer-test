import pytest

from datamodules.bcic4_2a import _get_2a_target_sessions


class _FakeSubjectDataset:
    def split(self, by):
        assert by == "session"
        return {"session_T": "T", "session_E": "E"}


def test_chronological_protocol_adapts_on_training_session():
    assert _get_2a_target_sessions(_FakeSubjectDataset(), "train") == ("T", "E")


def test_transductive_protocol_adapts_on_test_session():
    assert _get_2a_target_sessions(_FakeSubjectDataset(), "test") == ("E", "E")


def test_unknown_adaptation_session_is_rejected():
    with pytest.raises(ValueError):
        _get_2a_target_sessions(_FakeSubjectDataset(), "both")
