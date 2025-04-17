# NLP-CS4420
Contains assignments and project for the course Natural Language Processing - Theory and Applications (CS-4420) at Ashoka University in Spring 2025

## Assignment 1
POS Tagging, Named Entity Recognition, Coreference Resolution, Entity Resolution. Contains scripts for scraping Google, Bing, and MSN articles about ozempic, and performing the mentioned tasks on said articles. All tasks available in separate scripts with their respective CSV outputs. For manual annotation to measure accuracy of the NER and Coref model, I recommend using Label Studio. 

## Assignment 2
Classification of a mental health tweets dataset with LSTMs and Transformers, explanation generation with LIME and Occlusion, identification and interpretation of class signatures. Basic lstm architecture - embedding layer to convert words into dense vector representations, LSTM layer to capture textual patterns, fully connected (dense) layer to LSTM outputs to classification labels, softmax activation for producing probabilities for each mental health class. I've performed the same task with BERT for comparison, colab notebook attached here: https://colab.research.google.com/drive/12eIjTfEC0obRQ4ITrpGU5ZYrzaE8Jtto?usp=sharing

## Project - Emotion Aware Speech Transcription
Basically the title, uses a mix of transcription as well as audio based analyses to determine emotional context of a given audio recording. OpenAI Whisper for audio transcribing and translation, ffmpeg for extracting speech information such as MFCCs, phonemes, pitch info, etc.