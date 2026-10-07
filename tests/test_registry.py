import pytest

from openwhisprflow.config import SttConfig
from openwhisprflow.stt import registry
from openwhisprflow.stt.base import SttEngine, SttError


def test_available_lists_local_engines_first() -> None:
    infos = registry.available()
    assert infos[0].kind == "local"
    assert "parakeet" in [i.name for i in infos]
    assert all(i.kind in ("local", "cloud") for i in infos)


def test_create_parakeet_is_lazy_and_matches_contract() -> None:
    engine = registry.create(SttConfig(engine="parakeet"))
    assert isinstance(engine, SttEngine)
    assert engine.name == "parakeet" and engine.kind == "local" and engine.sample_rate == 16000


@pytest.mark.parametrize("name", ["nope", "../config", "base", "registry"])
def test_unknown_engine_raises_stt_error(name: str) -> None:
    with pytest.raises(SttError) as e:
        registry.create(SttConfig(engine=name))
    assert e.value.code == "engine_missing"


def test_missing_cloud_module_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    real = registry._module
    monkeypatch.setattr(registry, "_module", lambda n: None if n == "soniox" else real(n))
    assert "soniox" not in [i.name for i in registry.available()]


def test_is_available_hook_hides_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from openwhisprflow.stt import whisper

    monkeypatch.setattr(whisper, "is_available", lambda: False)
    assert "whisper" not in [i.name for i in registry.available()]
    with pytest.raises(SttError):
        registry.create(SttConfig(engine="whisper"))


def test_parakeet_variants_are_pinned() -> None:
    from openwhisprflow.stt import parakeet

    assert parakeet.INFO.default_model in parakeet.VARIANTS
    for v in parakeet.VARIANTS.values():
        assert len(v.revision) == 40
        assert set(v.files) == {"config.json", "vocab.txt", "encoder-model.int8.onnx",
                                "decoder_joint-model.int8.onnx"}
        assert all(len(sha) == 64 and size > 0 for _, size, sha in v.files.values())
    with pytest.raises(SttError) as e:
        parakeet.create(SttConfig(engine="parakeet", model="parakeet-9000"))
    assert e.value.code == "bad_model"


def test_parakeet_provider_selection() -> None:
    from openwhisprflow.stt.parakeet import select_providers

    cpu_only = ["CPUExecutionProvider"]
    gpu = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    assert select_providers("auto", cpu_only) == cpu_only
    assert select_providers("auto", gpu) == gpu
    assert select_providers("cpu", gpu) == cpu_only
    assert select_providers("directml", gpu) == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert select_providers("cuda", cpu_only) == cpu_only
