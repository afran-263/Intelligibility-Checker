# Intelligibility Checker
Evaluating Dysarthric Speech using Goodness of Pronunciation (GoP) and Wav2Vec2

## Project Overview
Intelligibility-Checker is a diagnostic system designed to assess the pronunciation quality of dysarthric speech. The project utilizes Wav2Vec2 for feature extraction, Montreal Forced Aligner (MFA) for alignment, and a neural network-based phoneme classifier to compute GoP scores and related uncertainty metrics for evaluating speech intelligibility.

## Key Features
- Wav2Vec2 Feature Extraction

- Montreal Forced Aligner (MFA) for phoneme-to-frame alignment

- Phoneme-Level GoP Computation using:

  - Entropy
  - Margin
  - MaxLogit
  - LogitMargin

- Dysarthria-Specific Evaluation

- Phoneme Classifier Training using Healthy Speech

- PER (Phoneme Error Rate) Calculation

## Outputs include:

- GoP score

- Entropy

- Margin

- MaxLogit

- LogitMargin
