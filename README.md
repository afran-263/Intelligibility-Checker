# Intelligibility Checker
A Web-Based Tool for Evaluating Pronunciation Using Wav2Vec2 and Word Error Rate (WER)

## Overview
Intelligibility-Checker is a web-based application that allows users to evaluate their spoken pronunciation by comparing their speech to a reference sentence using automatic speech recognition (ASR) powered by Wav2Vec2.0. It calculates the Word Error Rate (WER) to quantify how close the spoken sentence is to the reference.

## Features
- Real-Time Speech Recording from your microphone

- ASR Transcription using Facebook’s Wav2Vec2 model

- Word Error Rate (WER) computation with substitutions, insertions, and deletions

- Flask-based Web Interface

- Whisper-based optional transcription support

## Running the App
- Open your browser and go to http://127.0.0.1:5000/

- Enter a reference sentence

- The system will record your voice, transcribe it, and calculate WER

## Behind the Scenes
- Record_audio() captures 5 seconds of audio using your microphone.

- Transcribe() uses the pretrained facebook/wav2vec2-large-960h model for speech-to-text.

- The transcription is compared against the reference using a custom WER function.

- The system outputs the Predicted Text and Word Error Rate (WER) score.
