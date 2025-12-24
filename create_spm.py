import os
import torchaudio
import sentencepiece as spm

# [설정] run_conformer.sh에 설정한 DATA_PATH와 동일하게 설정하세요.
DATA_PATH = "/workspace/datasets"  # 예: "/mnt/large_disk/librispeech_data"
MODEL_PREFIX = "spm_librispeech_1024"
VOCAB_SIZE = 1024

def train_spm():
    print(f"Loading LibriSpeech dataset from {DATA_PATH}...")

    # 데이터셋이 없으면 다운로드합니다 (train-clean-100만 사용하여 토크나이저 학습)
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=DATA_PATH, 
        url="train-clean-100", 
        download=True
    )

    print("Extracting transcripts...")
    text_file = "librispeech_transcripts.txt"

    # 텍스트 파일로 추출
    with open(text_file, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            # (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
            transcript = dataset[i][2] 
            f.write(transcript + "\n")

    print(f"Training SentencePiece model (vocab_size={VOCAB_SIZE})...")

    # SentencePiece 학습
    # unigram 모델, vocab_size 1024, bos/eos/unk id 포함 설정
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

    print(f"Done! Created {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")

    # 임시 텍스트 파일 삭제 (선택사항)
    if os.path.exists(text_file):
        os.remove(text_file)

if __name__ == "__main__":
    train_spm()
