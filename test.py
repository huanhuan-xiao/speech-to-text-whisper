#!/usr/bin/python3
# -- coding: utf-8 --
# @Time : 2023/7/17 下午5:08
# @Author : huan_xhh
# @FileName: test.py

import whisper

model = whisper.load_model("/root/.cache/whisper/large-v2.pt")

# load audio and pad/trim it to fit 30 seconds
#audio = whisper.load_audio("/root/下载/play_digit.wav")
audio = whisper.load_audio("./sounds/play_digit.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
#options = whisper.DecodingOptions(fp16 = False, prompt="以下是普通话的句子",)  # 简体中文增加 prompt，原来默认的是下面的那句。
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
#print(result.text)
from zhconv import convert
print(convert(result.text, 'zh-cn'))
