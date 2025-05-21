import pytest
from livekit.plugins.openai import TTS
from livekit.plugins.openai.tts import TTSVoices, _RESPONSE_FORMATS # For type hinting if needed, actual values used directly
from livekit.agents.types import NOT_GIVEN

def test_tts_with_ollama_defaults():
    tts = TTS.with_ollama()

    assert tts._client.api_key == "ollama"
    assert str(tts._client.base_url) == "http://localhost:11434/v1/"
    assert tts._opts.model == "tts-server"
    assert tts._opts.voice == "alloy"  # Default voice for with_ollama
    assert tts._opts.speed == 1.0     # Default speed for with_ollama
    # Default from TTS constructor when not_given
    assert tts._opts.instructions is None 
    assert tts._opts.response_format == "mp3" # Default from TTS constructor

def test_tts_with_ollama_custom_args():
    custom_model = "custom-tts-model"
    custom_voice: TTSVoices = "nova" # Example of using the type
    custom_speed = 1.5
    custom_base_url = "http://custom.ollama.tts:12345/v1"
    custom_instructions = "Speak clearly and concisely."
    custom_response_format: _RESPONSE_FORMATS = "pcm"

    tts = TTS.with_ollama(
        model=custom_model,
        voice=custom_voice,
        speed=custom_speed,
        base_url=custom_base_url,
        instructions=custom_instructions,
        response_format=custom_response_format
    )

    assert tts._client.api_key == "ollama"
    assert str(tts._client.base_url) == f"{custom_base_url}/"
    assert tts._opts.model == custom_model
    assert tts._opts.voice == custom_voice
    assert tts._opts.speed == custom_speed
    assert tts._opts.instructions == custom_instructions
    assert tts._opts.response_format == custom_response_format
