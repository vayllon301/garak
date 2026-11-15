import pytest

from garak.probes.base import Probe


@pytest.mark.parametrize("intent_container", [list, tuple])
def test_pruning_isolates_aligned_class_data(monkeypatch, intent_container):
    """Pruning one instance preserves class data and later instances' alignment."""

    class ClassDataProbe(Probe):
        prompts = [f"prompt {i}" for i in range(4)]
        triggers = [f"trigger {i}" for i in range(4)]
        _prompt_intents = intent_container(f"intent {i}" for i in range(4))

    monkeypatch.setattr("garak.probes.base.random.sample", lambda population, k: [1, 3])
    capped = ClassDataProbe()
    capped._prune_data(2, prune_triggers=True)
    later = ClassDataProbe()

    assert capped.prompts == ["prompt 0", "prompt 2"], "selected prompts survive"
    assert capped.triggers == ["trigger 0", "trigger 2"], "triggers follow prompts"
    assert [capped._intent_for_seq(i) for i in range(2)] == [
        "intent 0",
        "intent 2",
    ], "intents follow the surviving prompts"
    assert len(ClassDataProbe.prompts) == 4, "class prompts remain complete"
    assert len(ClassDataProbe.triggers) == 4, "class triggers remain complete"
    assert list(ClassDataProbe._prompt_intents) == [
        f"intent {i}" for i in range(4)
    ], "class intents remain complete"
    assert later.prompts == [
        f"prompt {i}" for i in range(4)
    ], "later prompts are complete"
    assert [later._intent_for_seq(i) for i in range(4)] == [
        f"intent {i}" for i in range(4)
    ], "later instances retain the original intent alignment"
