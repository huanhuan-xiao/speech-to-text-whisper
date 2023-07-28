#!/usr/bin/python3
# -- coding: utf-8 --
# @Time : 2023/7/26 上午8:50
# @Author : huan_xhh
# @FileName: test1.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import librosa
import torch
from zhconv import convert
import warnings

warnings.filterwarnings("ignore")

audio_file = f"test.wav"
#load audio file
audio, sampling_rate = librosa.load(audio_file, sr=16_000)

# # audio
# display.Audio(audio_file, autoplay=True)

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
tokenizer = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

processor.save_pretrained("openai/model/whisper-large-v2")
model.save_pretrained("openai/model/whisper-large-v2")
tokenizer.save_pretrained("openai/model/whisper-large-v2")

processor = WhisperProcessor.from_pretrained("openai/model/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/model/whisper-large-v2")
tokenizer = WhisperProcessor.from_pretrained("openai/model/whisper-large-v2")


# load dummy dataset and read soundfiles
# ds = load_dataset("common_voice", "fr", split="test", streaming=True)
# ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
# input_speech = next(iter(ds))["audio"]["array"]
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
input_features = processor(audio, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
# transcription = processor.batch_decode(predicted_ids)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
print('转化为简体结果：', convert(transcription, 'zh-cn'))
