import speech_recognition as sr
import queue


# Queue to hold audio data
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def transcribe_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Adjusting for ambient noise... Please wait.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Ready! Start speaking.")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        print("Processing...")
        text = recognizer.recognize_google(audio)
        return text

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
    wer_result = ((S) + D + I) / float(N) if N > 0 else 0.0
    print(f"Substitutions: {S}, Deletions: {D}, Insertions: {I}")
    return wer_result

if __name__ == "__main__":
    reference_sentence = input("Enter the reference text: ")
    transcribed_text = transcribe_audio()
    print("Transcribed Text:", transcribed_text)
    wer_score = wer(reference_sentence, transcribed_text)
    print(f"Word Error Rate (WER): {wer_score:.2f}")
    if wer_score <= 0.4:  # You can adjust this threshold
        print("Sentence is intelligible.")
    else:
        print("Sentence is NOT intelligible.")