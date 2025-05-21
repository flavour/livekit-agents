import asyncio
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer

# Import the STT class from the plugin
# Adjust the import path if necessary based on how it's structured
from livekit.plugins.whisperx.stt import STT as WhisperXSTT
from livekit.plugins.whisperx.stt import STTOptions as WhisperXSTTOptions


# Minimal audio data for testing (e.g., 1 second of silence or simple tone)
# 48kHz, 16-bit mono, 1 second = 48000 samples * 2 bytes/sample = 96000 bytes
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2
DURATION_S = 1
SILENT_FRAME_DATA = np.zeros(SAMPLE_RATE * DURATION_S, dtype=np.int16).tobytes()

TEST_AUDIO_FRAME = rtc.AudioFrame(
    data=SILENT_FRAME_DATA,
    sample_rate=SAMPLE_RATE,
    num_channels=NUM_CHANNELS,
    samples_per_channel=SAMPLE_RATE * DURATION_S,
)


@pytest.fixture
def mock_whisperx_load_model():
    with patch("whisperx.load_model") as mock_load:
        mock_model_instance = MagicMock()
        # Mock the transcribe method
        mock_model_instance.transcribe = MagicMock(
            return_value={
                "segments": [
                    {
                        "text": "This is a test transcription.",
                        "start": 0.0,
                        "end": 1.0,
                        # other segment data if needed by the plugin
                    }
                ],
                "language": "en",
            }
        )
        mock_load.return_value = mock_model_instance
        yield mock_load


@pytest.fixture
def whisperx_stt(mock_whisperx_load_model):
    # Using a minimal configuration for testing
    return WhisperXSTT(model_size="tiny", device="cpu", compute_type="int8")


@pytest.mark.asyncio
async def test_recognize_non_streaming(whisperx_stt):
    buffer = AudioBuffer(
        data=TEST_AUDIO_FRAME.data,
        sample_rate=TEST_AUDIO_FRAME.sample_rate,
        num_channels=TEST_AUDIO_FRAME.num_channels,
    )

    event = await whisperx_stt.recognize(buffer=buffer)

    assert event is not None
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert len(event.alternatives) == 1
    assert event.alternatives[0].text == "This is a test transcription."
    assert event.alternatives[0].language == "en"

    # Check if whisperx.load_model was called (it is by the fixture)
    whisperx_stt._model.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_recognize_streaming(whisperx_stt):
    stream = whisperx_stt.stream()

    async def event_task():
        events = []
        async for event in stream:
            events.append(event)
        return events

    # Push a frame
    stream.push_frame(TEST_AUDIO_FRAME)
    # Mark the end of the segment
    stream.flush()
    # End the input to signal no more audio is coming
    stream.end_input() # This should trigger processing of any remaining audio in SpeechStream

    # Get events
    events = await asyncio.wait_for(event_task(), timeout=2.0) # Increased timeout slightly

    await stream.aclose()

    assert len(events) > 0, "Should have received at least one event"
    
    final_transcript_events = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(final_transcript_events) == 1, "Should have one final transcript event"
    
    event = final_transcript_events[0]
    assert len(event.alternatives) == 1
    assert event.alternatives[0].text == "This is a test transcription."
    assert event.alternatives[0].language == "en"

    # Check if transcribe was called (it should be by _process_accumulated_audio)
    whisperx_stt._model.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_multiple_frames(whisperx_stt):
    stream = whisperx_stt.stream()

    async def event_collector():
        return [event async for event in stream]

    results_task = asyncio.create_task(event_collector())

    # Push multiple frames
    for _ in range(3):
        stream.push_frame(TEST_AUDIO_FRAME)
        await asyncio.sleep(0.01) # Simulate slight delay between frames

    stream.flush() # Process the first batch
    
    # Reset the mock for the second call
    # The way the stream processes, transcribe might be called per flush
    # For this test, let's assume one transcribe call per flush that has data.
    # The fixture's model is shared, so we need to handle mock call counts carefully.
    # If the stream's _process_accumulated_audio calls the STT's _recognize_impl,
    # and _recognize_impl calls self._model.transcribe, then the mock on self._model
    # will be hit.

    # Let's refine the mock assertion. The mock is on the model instance within whisperx_stt.
    # Each call to _process_accumulated_audio will call _recognize_impl, which calls transcribe.
    
    # Since the mock is on the instance, it will accumulate calls.
    # First flush:
    await asyncio.sleep(0.2) # Allow time for processing the first flush

    # Push more frames
    for _ in range(2):
        stream.push_frame(TEST_AUDIO_FRAME)
        await asyncio.sleep(0.01)

    stream.end_input() # Process the second batch and close

    events = await asyncio.wait_for(results_task, timeout=5.0)
    await stream.aclose()

    final_transcripts = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(final_transcripts) == 2, "Should have two final transcript events for two flushes"
    
    for event in final_transcripts:
        assert len(event.alternatives) == 1
        assert event.alternatives[0].text == "This is a test transcription."

    assert whisperx_stt._model.transcribe.call_count == 2


@pytest.mark.asyncio
async def test_streaming_no_audio_then_flush(whisperx_stt):
    stream = whisperx_stt.stream()

    async def event_collector():
        return [event async for event in stream]

    results_task = asyncio.create_task(event_collector())

    stream.flush() # Flush with no audio
    stream.end_input()

    events = await asyncio.wait_for(results_task, timeout=2.0)
    await stream.aclose()
    
    final_transcripts = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(final_transcripts) == 0, "Should not produce transcripts if no audio was pushed"
    whisperx_stt._model.transcribe.assert_not_called()
