import itertools
import json
import logging
import os
import zlib

from dataclasses import asdict, dataclass
from inspect import signature
from math import ceil
from typing import BinaryIO, Iterable, List, Optional, Tuple, Union
from warnings import warn

import ctranslate2
import numpy as np
import tokenizers

from tqdm import tqdm

# faster-whisper의 기존 모듈들을 그대로 사용합니다.
from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import download_model, format_timestamp, get_end, get_logger
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
    merge_segments,
)

from faster_whisper.transcribe import Word, Segment, TranscriptionOptions, TranscriptionInfo
from faster_whisper.transcribe import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import download_model, format_timestamp, get_end, get_logger, get_compression_ratio, get_suppressed_tokens


class HybridInferencePipeline:
    """
    A hybrid model that combines the speed of BatchedInferencePipeline for independent
    chunks and the contextual accuracy of WhisperModel for sequential chunks.
    """

    def __init__(
        self,
        model,
    ):
        """
        Initializes the HybridWhisperModel.
        """
        self.model: WhisperModel = model
        self.logger = self.model.logger

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        multilingual: bool = False,
        vad_filter: bool = True,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = 30,
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        hotwords: Optional[str] = None,
        language_detection_threshold: Optional[float] = 0.5,
        language_detection_segments: int = 1,
        batch_size: int = 8,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """
        Transcribes an audio file using a hybrid approach.
        """
        sampling_rate = self.whisper_model.feature_extractor.sampling_rate

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)
        
        duration = audio.shape[0] / sampling_rate
        self.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )

        if not vad_filter:
            self.logger.warning("VAD filter is disabled. The model will process the audio sequentially.")
            return self.whisper_model.transcribe(
                audio, language=language, task=task, log_progress=log_progress,
                beam_size=beam_size, best_of=best_of, patience=patience, length_penalty=length_penalty,
                repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=temperature, compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold, no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                prompt_reset_on_temperature=prompt_reset_on_temperature,
                initial_prompt=initial_prompt, prefix=prefix, suppress_blank=suppress_blank,
                suppress_tokens=suppress_tokens, without_timestamps=without_timestamps,
                max_initial_timestamp=max_initial_timestamp, word_timestamps=word_timestamps,
                prepend_punctuations=prepend_punctuations, append_punctuations=append_punctuations,
                multilingual=multilingual, vad_filter=False,
                vad_parameters=vad_parameters, max_new_tokens=max_new_tokens,
                chunk_length=chunk_length, clip_timestamps=clip_timestamps,
                hallucination_silence_threshold=hallucination_silence_threshold,
                hotwords=hotwords, language_detection_threshold=language_detection_threshold,
                language_detection_segments=language_detection_segments
            )

        if vad_parameters is None:
            vad_parameters = VadOptions(max_speech_duration_s=chunk_length)
        elif isinstance(vad_parameters, dict):
            vad_parameters.pop("max_speech_duration_s", None)
            vad_parameters = VadOptions(max_speech_duration_s=chunk_length, **vad_parameters)
        
        speech_ts = get_speech_timestamps(audio, vad_parameters)
        merged_chunks_info = merge_segments(speech_ts, vad_parameters)

        if not merged_chunks_info:
            return [], self._create_empty_info(duration, vad_parameters, language)

        duration_after_vad = sum(
            (chunk["end"] - chunk["start"]) for chunk in merged_chunks_info
        ) / sampling_rate
        
        self.logger.info(
            "VAD filter removed %s of audio",
            format_timestamp(duration - duration_after_vad),
        )

        is_artificial_split = [False] * len(merged_chunks_info)
        for i in range(1, len(merged_chunks_info)):
            if merged_chunks_info[i]['start'] == merged_chunks_info[i-1]['end']:
                is_artificial_split[i] = True

        processing_groups = []
        current_group = []
        for i, chunk_info in enumerate(merged_chunks_info):
            current_group.append(chunk_info)
            if i == len(merged_chunks_info) - 1 or not is_artificial_split[i+1]:
                is_seq = len(current_group) > 1 or (is_artificial_split[i] if i > 0 else False)
                group_type = "sequential" if is_seq else "independent"
                processing_groups.append({"type": group_type, "chunks_info": current_group})
                current_group = []

        all_segments = []
        independent_audio_chunks = []
        independent_chunks_metadata = []
        
        first_chunk_audio, _ = collect_chunks(audio, [merged_chunks_info[0]])
        language, language_probability, all_language_probs = self._detect_language(
            language, first_chunk_audio[0]
        )

        tokenizer = Tokenizer(
            self.whisper_model.hf_tokenizer,
            self.whisper_model.model.is_multilingual,
            task=task,
            language=language,
        )

        # Batch 처리에 사용할 옵션 객체 (suppress_tokens는 미리 처리)
        temperatures_option = list(temperature) if isinstance(temperature, (list, tuple)) else [temperature]
        suppress_tokens_processed = get_suppressed_tokens(tokenizer, suppress_tokens)
        
        options = TranscriptionOptions(
            beam_size=beam_size, best_of=best_of, patience=patience, length_penalty=length_penalty,
            repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold, no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            temperatures=temperatures_option,
            initial_prompt=initial_prompt, prefix=prefix,
            suppress_blank=suppress_blank, suppress_tokens=suppress_tokens_processed,
            without_timestamps=without_timestamps, max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps, prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations, multilingual=multilingual,
            max_new_tokens=max_new_tokens, clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold, hotwords=hotwords,
        )
        
        pbar = tqdm(total=len(merged_chunks_info), unit="chunks", disable=not log_progress)

        for group in processing_groups:
            if group["type"] == "sequential":
                self._process_independent_chunks(
                    independent_audio_chunks, independent_chunks_metadata, all_segments,
                    tokenizer, options, batch_size, pbar, sampling_rate
                )
                independent_audio_chunks, independent_chunks_metadata = [], []

                group_audio, _ = collect_chunks(audio, group["chunks_info"])
                concatenated_audio = np.concatenate(group_audio)
                
                current_initial_prompt = initial_prompt
                if all_segments and condition_on_previous_text:
                    current_initial_prompt = tokenizer.decode(all_segments[-1].tokens)
                
                # 순차 처리 시에는 원본 suppress_tokens를 전달
                seq_segments, _ = self.whisper_model.transcribe(
                    concatenated_audio, language=language, task=task, log_progress=False,
                    beam_size=beam_size, best_of=best_of, patience=patience, length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                    temperature=temperature, compression_ratio_threshold=compression_ratio_threshold,
                    log_prob_threshold=log_prob_threshold, no_speech_threshold=no_speech_threshold,
                    condition_on_previous_text=condition_on_previous_text,
                    prompt_reset_on_temperature=prompt_reset_on_temperature,
                    initial_prompt=current_initial_prompt, prefix=prefix, suppress_blank=suppress_blank,
                    suppress_tokens=suppress_tokens,  # 원본 리스트 전달
                    without_timestamps=without_timestamps,
                    max_initial_timestamp=max_initial_timestamp, word_timestamps=word_timestamps,
                    prepend_punctuations=prepend_punctuations, append_punctuations=append_punctuations,
                    multilingual=multilingual, vad_filter=False,
                    max_new_tokens=max_new_tokens, 
                    hallucination_silence_threshold=hallucination_silence_threshold,
                    hotwords=hotwords
                )

                time_offset = group["chunks_info"][0]["start"] / sampling_rate
                for seg in seq_segments:
                    seg.start += time_offset
                    seg.end += time_offset
                    if seg.words:
                        for word in seg.words:
                            word.start += time_offset
                            word.end += time_offset
                    all_segments.append(seg)
                pbar.update(len(group["chunks_info"]))

            else:
                audio_chunk, metadata = collect_chunks(audio, group["chunks_info"])
                independent_audio_chunks.extend(audio_chunk)
                independent_chunks_metadata.extend(metadata)
        
        self._process_independent_chunks(
            independent_audio_chunks, independent_chunks_metadata, all_segments,
            tokenizer, options, batch_size, pbar, sampling_rate
        )

        pbar.close()

        all_segments.sort(key=lambda s: s.start)
        for i, seg in enumerate(all_segments):
            seg.id = i + 1

        info = TranscriptionInfo(
            language=language, language_probability=language_probability,
            duration=duration, duration_after_vad=duration_after_vad,
            all_language_probs=all_language_probs,
            transcription_options=options,
            vad_options=vad_parameters,
        )

        return (seg for seg in all_segments), info

    def _process_independent_chunks(self, audio_chunks, chunks_metadata, all_segments, tokenizer, options, batch_size, pbar, sampling_rate):
        if not audio_chunks:
            return

        features = [self.whisper_model.feature_extractor(chunk) for chunk in audio_chunks]
        padded_features = np.stack([pad_or_trim(feature, self.whisper_model.feature_extractor.nb_max_frames) for feature in features])
        
        for i in range(0, len(padded_features), batch_size):
            batch_features = padded_features[i : i + batch_size]
            batch_metadata = chunks_metadata[i : i + batch_size]
            
            encoder_output = self.whisper_model.encode(batch_features)
            
            results = self.whisper_model.model.generate(
                encoder_output,
                [self.whisper_model.get_prompt(tokenizer, [], without_timestamps=options.without_timestamps) for _ in range(len(batch_features))],
                beam_size=options.beam_size, patience=options.patience,
                length_penalty=options.length_penalty, repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size, suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens, max_length=self.whisper_model.max_length,
                return_scores=True, return_no_speech_prob=True,
            )

            batch_subsegments = []
            batch_avg_logprobs = []
            batch_no_speech_probs = []

            for result_idx, result in enumerate(results):
                tokens = result.sequences_ids[0]
                seq_len = len(tokens)
                avg_logprob = result.scores[0] * (seq_len**options.length_penalty) / (seq_len + 1) if seq_len > 0 else 0
                
                current_metadata = batch_metadata[result_idx]
                time_offset = current_metadata["start_time"] # start_time (초) 사용
                segment_duration = current_metadata["end_time"] - current_metadata["start_time"]
                seek_in_frames = int(time_offset * self.whisper_model.frames_per_second)

                subsegments, _, _ = self.whisper_model._split_segments_by_timestamps(
                    tokenizer, tokens, time_offset, 
                    batch_features[result_idx].shape[-1], 
                    segment_duration,
                    seek=seek_in_frames
                )
                batch_subsegments.append(subsegments)
                batch_avg_logprobs.append(avg_logprob)
                batch_no_speech_probs.append(result.no_speech_prob)
            
            if options.word_timestamps and any(batch_subsegments):
                # add_word_timestamps는 segment의 리스트를 받으므로, 2차원 리스트를 전달
                self.whisper_model.add_word_timestamps(
                    batch_subsegments, tokenizer, encoder_output,
                    [f.shape[-1] for f in batch_features],
                    options.prepend_punctuations, options.append_punctuations,
                    last_speech_timestamp=0
                )
            
            for idx, subsegment_list in enumerate(batch_subsegments):
                for sub in subsegment_list:
                    text = tokenizer.decode(sub["tokens"])
                    if not text.strip():
                        continue
                    
                    compression_ratio = get_compression_ratio(text)
                    all_segments.append(
                        Segment(
                            id=0, seek=sub["seek"],
                            start=round(sub["start"], 3), end=round(sub["end"], 3),
                            text=text, tokens=sub["tokens"],
                            avg_logprob=batch_avg_logprobs[idx],
                            compression_ratio=compression_ratio,
                            no_speech_prob=batch_no_speech_probs[idx],
                            words=[Word(**word) for word in sub["words"]] if sub.get("words") else None,
                            temperature=options.temperatures[0],
                        )
                    )
            pbar.update(len(batch_features))

    def _detect_language(self, language, audio_chunk):
        if language is not None:
            return language, 1.0, None
        
        if not self.whisper_model.model.is_multilingual:
            return "en", 1.0, None

        features = self.whisper_model.feature_extractor(audio_chunk)
        lang, prob, all_probs = self.whisper_model.detect_language(features=features)
        self.logger.info(
            "Detected language '%s' with probability %.2f", lang, prob
        )
        return lang, prob, all_probs

    def _create_empty_info(self, duration, vad_parameters, language):
        return TranscriptionInfo(
            language=language or "en",
            language_probability=1.0,
            duration=duration,
            duration_after_vad=0,
            all_language_probs=None,
            transcription_options=None,
            vad_options=vad_parameters,
        )