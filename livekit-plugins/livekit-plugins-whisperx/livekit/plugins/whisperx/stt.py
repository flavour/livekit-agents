from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch # WhisperX dependency
import whisperx # WhisperX dependency

from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer

from .version import __version__

# Logging
# from livekit.agents import log
# logger = log.LoggingContext(name=__name__)
# TODO: Add proper logging context if needed


@dataclass
class STTOptions:
    language: str | None = None  # WhisperX can auto-detect
    model_size: str = "base"  # Default model size
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "int8" # or "int8" for CPU
    batch_size: int = 16
    hf_token: str | None = None # For gated models like whisper-large-v3
    # Add other WhisperX specific options here as needed
    # e.g., vad_options, align_model, diarize, etc.


class STT(stt.STT[stt.SpeechEvent]):
    def __init__(
        self,
        *,
        model_size: str = "base",
        language: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        batch_size: int = 16,
        hf_token: str | None = None,
        # Add other WhisperX specific init arguments
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, # WhisperX can support streaming-like behavior
                interim_results=False, # WhisperX typically provides final results
            )
        )
        
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _compute_type = compute_type or ("float16" if _device == "cuda" else "int8")

        self._opts = STTOptions(
            language=language,
            model_size=model_size,
            device=_device,
            compute_type=_compute_type,
            batch_size=batch_size,
            hf_token=hf_token,
        )

        # Load WhisperX model
        # Models are typically downloaded on first use by whisperx
        # logger.info(f"Loading WhisperX model: {self._opts.model_size} on device: {self._opts.device} with compute_type: {self._opts.compute_type}")
        try:
            self._model = whisperx.load_model(
                self._opts.model_size,
                device=self._opts.device,
                compute_type=self._opts.compute_type,
                language=self._opts.language, # Can be None for auto-detect
                asr_options={"initial_prompt": None}, # Add other asr_options if needed
                # download_root="path/to/your/models" # Optional: if you want to specify model download location
            )
            # logger.info("WhisperX model loaded successfully.")
        except Exception as e:
            # logger.error(f"Failed to load WhisperX model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load WhisperX model: {e}") from e
            
        # Optionally load alignment model if needed for word timings
        # try:
        #     if self._opts.language: # Alignment model requires language
        #          self._align_model, self._align_metadata = whisperx.load_align_model(language_code=self._opts.language, device=self._opts.device)
        #          # logger.info("WhisperX alignment model loaded successfully.")
        #     else:
        #          self._align_model = None
        #          # logger.info("WhisperX alignment model not loaded as language is not specified for STTOptions.")
        # except Exception as e:
        #     # logger.warning(f"Failed to load WhisperX alignment model: {e}", exc_info=True)
        #     self._align_model = None # Continue without alignment if it fails


    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions, # Not directly used by local WhisperX
    ) -> stt.SpeechEvent:
        # Convert AudioBuffer to a format WhisperX understands (e.g., numpy array)
        # AudioBuffer contains raw audio data (PCM, 16-bit int, 48kHz, mono)
        # WhisperX expects a filepath or a numpy array (float32)
        
        # logger.debug(f"Recognizing audio buffer of duration: {buffer.duration}s")

        audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe
        try:
            # logger.debug(f"Starting transcription with WhisperX model: {self._opts.model_size}")
            result = self._model.transcribe(
                audio_data,
                batch_size=self._opts.batch_size,
                language=language if is_given(language) else self._opts.language, # Override if language is provided
                # chunk_size=, # for longer audio files
                # print_progress=False,
                # combined_progress=False
            )
            # logger.debug(f"WhisperX transcription result: {result}")
        except Exception as e:
            # logger.error(f"WhisperX transcription failed: {e}", exc_info=True)
            # Consider raising a specific STTError or APIError if appropriate
            raise RuntimeError(f"WhisperX transcription failed: {e}") from e

        alternatives = []
        if result and "segments" in result:
            full_text = " ".join([seg["text"].strip() for seg in result["segments"]]).strip()
            if full_text:
                # WhisperX provides segments with start/end times.
                # For a single final transcript, we can concatenate them.
                # Or, if word timings are needed, align_model should be used.
                
                # For now, create one SpeechData with the full text.
                # Language detection might be available in `result["language"]`
                detected_language = result.get("language", self._opts.language or "unknown")

                # Word timings if alignment model is loaded and successful
                # if self._align_model and self._align_metadata and result["segments"]:
                #    try:
                #        aligned_result = whisperx.align(result["segments"], self._align_model, self._align_metadata, audio_data, device=self._opts.device)
                #        # logger.debug(f"WhisperX alignment result: {aligned_result}")
                #        # Process aligned_result to get word timings if needed
                #        # For now, we'll stick to segment level timings or full text
                #        # Example: sd.words = [stt.Word(text=w['word'], start_time=w['start'], end_time=w['end'], confidence=w.get('score', 0.0)) for w in aligned_segment['words']]
                #    except Exception as e:
                #        # logger.warning(f"WhisperX alignment failed: {e}", exc_info=True)
                #        pass # Proceed without alignment if it fails

                # Use segment start/end for overall timing if available and meaningful
                start_time = result["segments"][0]["start"] if result["segments"] else 0.0
                end_time = result["segments"][-1]["end"] if result["segments"] else buffer.duration

                alternatives.append(
                    stt.SpeechData(
                        language=detected_language,
                        text=full_text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=1.0,  # WhisperX doesn't always provide a single confidence score for the whole transcript
                    )
                )
        
        # logger.info(f"Transcription complete. Text: {alternatives[0].text if alternatives else 'No speech detected.'}")
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=alternatives,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        # logger.debug(f"Creating STT stream with language: {language if is_given(language) else self._opts.language}")
        # Options for the stream can be derived from the main STT options
        # or overridden if necessary
        stream_opts = STTOptions(
            language=language if is_given(language) else self._opts.language,
            model_size=self._opts.model_size,
            device=self._opts.device,
            compute_type=self._opts.compute_type,
            batch_size=self._opts.batch_size,
            hf_token=self._opts.hf_token,
        )
        return SpeechStream(stt_engine=self, opts=stream_opts, conn_options=conn_options)


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt_engine: STT, # Reference to the parent STT engine
        opts: STTOptions,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt_engine, conn_options=conn_options, sample_rate=48000) # LiveKit AudioFrames are 48kHz
        self._stt_engine = stt_engine
        self._opts = opts
        self._audio_buffer = AudioBuffer(sample_rate=48000, num_channels=1) # Accumulate audio here
        self._processing_task: asyncio.Task | None = None
        self._closed = False

        # WhisperX doesn't have a native "streaming" API in the same way as cloud services.
        # We need to simulate it by accumulating audio and processing it in chunks,
        # or by processing it when `flush` or `end_input` is called.
        # For true VAD-based streaming, whisperx.VadPipeline could be used,
        # but that's more complex to integrate directly into this STT class structure.
        # A simpler approach for now: process audio on flush/end_input.
        
        # logger.debug(f"SpeechStream initialized with options: {opts}")

    async def _run(self) -> None:
        # This method is called by the base class to start the stream processing.
        # It needs to handle audio pushed via `self._input_ch`
        # logger.debug("SpeechStream _run started.")
        try:
            while True:
                try:
                    item = await asyncio.wait_for(self._input_ch.recv(), timeout=0.1)  # Short timeout to allow checking _closed
                except asyncio.TimeoutError:
                    if self._closed and self._input_ch.empty():
                        # logger.debug("SpeechStream closing as stream is closed and input channel is empty.")
                        break 
                    continue # Continue waiting for input if not closed

                if isinstance(item, rtc.AudioFrame):
                    # logger.debug(f"SpeechStream received AudioFrame of duration {item.duration}s")
                    self._audio_buffer.write(item.data) # AudioBuffer expects bytes
                elif isinstance(item, self._FlushSentinel):
                    # logger.debug("SpeechStream received FlushSentinel.")
                    if self._audio_buffer.duration > 0.1: # Process if there's enough audio
                        await self._process_accumulated_audio()
                    # After processing, a SpeechEvent (FINAL_TRANSCRIPT) should have been sent.
                    # For streaming, we might want to send interim results or segment results.
                    # For now, we'll send a final transcript for the flushed segment.
                
                if self._input_ch.closed() and self._input_ch.empty():
                    # logger.debug("SpeechStream input channel closed and empty, processing remaining audio.")
                    if self._audio_buffer.duration > 0:
                         await self._process_accumulated_audio()
                    break # Exit loop when input channel is closed and drained

        except Exception as e:
            # logger.error(f"Error in SpeechStream _run: {e}", exc_info=True)
            # Emit an error event if appropriate
            # self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.ERROR, error=e)) # Example error event
            pass # Let the base class handle APIError propagation if it occurs in _process_accumulated_audio
        finally:
            # logger.debug("SpeechStream _run finished.")
            self._event_ch.close()


    async def _process_accumulated_audio(self) -> None:
        # logger.debug(f"Processing accumulated audio of duration: {self._audio_buffer.duration}s")
        if self._audio_buffer.is_empty:
            # logger.debug("No audio to process.")
            return

        # Use the parent STT engine's _recognize_impl for actual transcription
        # This might need to be adapted if _recognize_impl is strictly for non-streaming
        # and expects a full buffer. For streaming, we're processing chunks.
        
        # Create a temporary AudioBuffer from the accumulated data for _recognize_impl
        # Important: _recognize_impl expects a complete AudioBuffer.
        # We need to ensure that the way we are calling it is consistent with its design.
        
        # The current _recognize_impl is designed for a single complete buffer.
        # For streaming, we are essentially calling it multiple times with chunks.
        # This might not be the most efficient way for WhisperX if it can handle
        # continuous streams better with its pipeline/chunking options internally.
        # However, for this structure, it's a viable approach.

        current_buffer_data = self._audio_buffer.read_all() # Get all data as bytes
        if not current_buffer_data:
            # logger.debug("No data in audio buffer to process.")
            return

        # Create a new AudioBuffer instance for this chunk
        # Note: This assumes AudioBuffer can be created directly from bytes if needed,
        # or that we construct it appropriately. The current AudioBuffer.data is a property.
        # Let's assume we can reconstruct it or pass the necessary parts.
        # The base AudioBuffer is created with sample_rate and num_channels.
        # When we call `write`, it appends. `read_all` gives us the bytes.
        # We need to make a *new* AudioBuffer that _recognize_impl can use.
        
        # Hacky way to make a new buffer for _recognize_impl:
        # This is not ideal as AudioBuffer is not designed to be easily reconstructed from raw bytes this way.
        # A better way would be if AudioBuffer had a method to "slice" or "copy" a segment.
        # For now, let's assume _recognize_impl can work with the raw audio data directly if we adapt it,
        # or we pass the numpy array.
        
        # Let's stick to the _recognize_impl as is, which means it needs an AudioBuffer.
        # This means our streaming SpeechStream needs to manage AudioBuffer instances carefully.

        # The simplest way: create a new AudioBuffer for each chunk.
        chunk_buffer = AudioBuffer(sample_rate=self._audio_buffer.sample_rate, num_channels=self._audio_buffer.num_channels)
        chunk_buffer.write(current_buffer_data)

        try:
            # logger.debug(f"Calling _recognize_impl for chunk of duration {chunk_buffer.duration}s")
            event = await self._stt_engine._recognize_impl(
                chunk_buffer,
                language=self._opts.language, # Use stream's language option
                conn_options=self._conn_options, # Pass along connection options
            )
            
            if event and event.alternatives:
                # logger.info(f"Stream processing produced transcript: {event.alternatives[0].text}")
                # Adapt start/end times if necessary, relative to the stream
                # For now, WhisperX provides absolute times within the chunk.
                # If we want times relative to the start of the whole stream, more complex tracking is needed.
                self._event_ch.send_nowait(event) 
            else:
                # logger.debug("Stream processing did not produce a transcript for the chunk.")
                # Optionally send an empty event or nothing
                pass
        except Exception as e:
            # logger.error(f"Error during stream's _recognize_impl call: {e}", exc_info=True)
            # How to signal this error to the consumer of the stream?
            # The base class RecognizeStream has error handling in _main_task for APIError.
            # If this is not an APIError, it might not be caught and retried by the base.
            # We might need to wrap it in an APIError or STTError if we want consistent error handling.
            # For now, let it propagate if it's a severe error.
            # Consider emitting a specific error event on self._event_ch
            # e.g., self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.ERROR, error_message=str(e)))
            raise # Re-raise to be caught by the main loop if it's an APIError
        finally:
            self._audio_buffer.clear() # Clear the buffer after processing the chunk


    async def aclose(self) -> None:
        # logger.debug("SpeechStream aclose called.")
        self._closed = True
        if self._input_ch.closed(): # If already closed, ensure task is awaited
             if self._task and not self._task.done():
                try:
                    await asyncio.wait_for(self._task, timeout=5.0) # Wait for task to finish
                except asyncio.TimeoutError:
                    # logger.warning("Timeout waiting for SpeechStream task to complete on aclose.")
                    pass # Task might be stuck, but we need to proceed with closing
        else:
            self._input_ch.close() # Signal to _run loop that input is finished
            # The _run loop should then process remaining audio and exit.
            # Base class aclose will cancel the task.
        
        # Ensure any final processing happens if there's data
        if self._audio_buffer and self._audio_buffer.duration > 0 and not self._input_ch.closed():
            # This case should ideally be handled by flush/end_input before aclose,
            # but as a safeguard:
            # logger.debug("Processing remaining audio in aclose.")
            # await self._process_accumulated_audio() # This might be problematic if called after task is cancelled.
            pass

        await super().aclose() # Calls base class aclose which cancels self._task
        # logger.debug("SpeechStream aclose finished.")
