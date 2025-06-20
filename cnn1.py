import torch
import torchaudio
import sounddevice as sd
import scipy.io.wavfile as wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from g2p_en import G2p
from jiwer import wer
import difflib
import nltk
import torchaudio.transforms as T
import whisper

# Download necessary NLTK data files
nltk.download('averaged_perceptron_tagger')

# Load models
print("🔄 Loading Wav2Vec2.0 model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model.eval()

g2p = G2p()

# Settings
SAMPLE_RATE = 16000
DURATION = 5  # seconds

def record_audio(duration=DURATION, filename="recorded.wav"):
    print(f"\n🎙️ Recording {duration} seconds of audio...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(filename, SAMPLE_RATE, recording)
    print("✅ Audio recorded.")
    return filename

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    return waveform.squeeze()

def transcribe(audio_tensor):
    inputs = processor(audio_tensor, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].lower()

def transcribe_with_beam_search(audio_tensor):
    print("hello")
    inputs = processor(audio_tensor, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    return tokenizer.decode(pred_ids[0], output_word_offsets=True)

def phonemize(text):
    print("hello1")
    phonemes = g2p(text)
    return [p for p in phonemes if p.isalpha()]

def denoise_audio(waveform):
    # Example: Apply a noise reduction filter
    print("hello2")
    transform = T.SpectralGate()
    return transform(waveform)

def normalize_audio(waveform):
    print("hello3")
    return waveform / waveform.abs().max()

def clean_transcription(text):
    # Remove extra spaces, punctuation, etc.
    print("hello4")
    return " ".join(text.split())

def transcribe_with_whisper(audio_file):
    print("hello5")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file, language="en")
    return result["text"]

def wer(reference, hypothesis):
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    N = len(ref_words)
    # Initialize matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost  # substitution
            )
    # Backtrack to find S, D, I
    i, j = len(ref_words), len(hyp_words)
    S, D, I = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]):
            if ref_words[i - 1] != hyp_words[j - 1]:
                S += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    wer_result = (S + D + I) / float(N) if N > 0 else 0.0
    print(f"Substitutions: {S}, Deletions: {D}, Insertions: {I}")
    return wer_result

if __name__ == "__main__":
    reference_sentence = input("Enter the reference text: ")
    audio_file = record_audio()
    audio_tensor = load_audio(audio_file)
    transcribed_text = transcribe(audio_tensor)
    print("Transcribed Text:", transcribed_text)
    wer_score = wer(reference_sentence, transcribed_text)
    print(f"Word Error Rate (WER): {wer_score:.2f}")