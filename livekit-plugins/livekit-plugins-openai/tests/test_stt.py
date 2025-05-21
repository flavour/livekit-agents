import pytest
from livekit.plugins.openai import STT
from livekit.agents.types import NOT_GIVEN

def test_stt_with_ollama_defaults():
    stt = STT.with_ollama()

    assert stt._client.api_key == "ollama"
    assert str(stt._client.base_url) == "http://localhost:11434/v1/"
    assert stt._opts.model == "whisper-large-v3"
    assert stt.capabilities.streaming is False
    # Default language from STT constructor when detect_language is False and no language is passed
    assert stt._opts.language == "en"
    assert stt._opts.detect_language is False
    assert stt._opts.prompt is NOT_GIVEN

def test_stt_with_ollama_custom_args():
    custom_model = "custom-stt-model"
    custom_base_url = "http://custom.ollama.url:12345/v1"
    custom_language = "fr"
    custom_prompt = "Bonjour le monde."

    stt = STT.with_ollama(
        model=custom_model,
        base_url=custom_base_url,
        language=custom_language,
        detect_language=False, # Explicitly set to False
        prompt=custom_prompt
    )

    assert stt._client.api_key == "ollama"
    assert str(stt._client.base_url) == f"{custom_base_url}/"
    assert stt._opts.model == custom_model
    assert stt._opts.language == custom_language
    assert stt._opts.detect_language is False
    assert stt._opts.prompt == custom_prompt
    assert stt.capabilities.streaming is False

def test_stt_with_ollama_detect_language_true():
    stt = STT.with_ollama(
        detect_language=True,
        language="this should be ignored" 
    )

    assert stt._client.api_key == "ollama"
    assert str(stt._client.base_url) == "http://localhost:11434/v1/"
    assert stt._opts.model == "whisper-large-v3"
    assert stt._opts.language == ""  # Language should be empty when detect_language is True
    assert stt._opts.detect_language is True
    assert stt.capabilities.streaming is False
