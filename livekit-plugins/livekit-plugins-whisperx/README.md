# WhisperX Plugin for LiveKit Agents

Provides speech-to-text (STT) functionality using [WhisperX](https://github.com/m-bain/whisperX).

## Installation

```bash
pip install livekit-plugins-whisperx
```

## Prerequisites

Ensure you have the necessary dependencies for WhisperX installed, including `ffmpeg` and a suitable PyTorch version compatible with your hardware (CPU or GPU). Refer to the [WhisperX documentation](https://github.com/m-bain/whisperX#installation) for detailed installation instructions.

You may also need to download Whisper models compatible with WhisperX.

## Usage

```python
from livekit.agents import stt
from livekit.plugins import whisperx

# Initialize WhisperX STT
stt_plugin = whisperx.STT()

# Example usage (non-streaming)
# async def transcribe_audio_file(filepath: str):
#     buffer = AudioBuffer.from_file(filepath)
#     event = await stt_plugin.recognize(buffer=buffer)
#     if event.alternatives:
#         print(f"Transcription: {event.alternatives[0].text}")

# Example usage (streaming)
# async def transcribe_stream(stream: rtc.AudioStream):
#     stt_stream = stt_plugin.stream()
#     async for frame in stream:
#         stt_stream.push_frame(frame)
#     await stt_stream.aclose()

#     async for event in stt_stream:
#         if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
#             print(f"Transcription: {event.alternatives[0].text}")
```

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for more information on using LiveKit Agents and plugins.
