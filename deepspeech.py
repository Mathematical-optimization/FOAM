import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# ==========================================
# 1. Configuration & Constants
# ==========================================
class Config:
    # Audio Params
    sample_rate = 16000
    n_fft = 400       # 25ms
    hop_length = 160  # 10ms
    n_mels = 80       # AlgoPerf default for LibriSpeech
    
    # Training Params
    batch_size = 8    # 메모리에 맞춰 조정
    learning_rate = 3e-4
    epochs = 1
    
    # Data Check
    max_audio_length = 320000 # [cite: 1887] Eliminate examples > 320k
    
    # Path
    data_path = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Simple Tokenizer (Instead of SentencePiece)
# ==========================================
class TextTransform:
    """
    논문은 SentencePiece(Vocab 1024)를 사용하지만[cite: 1887],
    실행 가능한 데모를 위해 간단한 Character-level Tokenizer로 구현했습니다.
    """
    def __init__(self):
        self.char_map = {"'": 0, " ": 1}
        self.index_map = {0: "'", 1: " "}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", start=2):
            self.char_map[c] = i
            self.index_map[i] = c
            
    def text_to_int(self, text):
        targets = []
        for c in text.lower():
            if c in self.char_map:
                targets.append(self.char_map[c])
        return torch.tensor(targets, dtype=torch.long)

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map.get(i.item(), ""))
        return "".join(string)

    def __len__(self):
        return len(self.char_map) + 1 # +1 for CTC blank

text_transform = TextTransform()

# ==========================================
# 3. Dataset & Preprocessing 
# ==========================================
class AlgoPerfLibriSpeech(Dataset):
    def __init__(self, root, url="train-clean-100", download=True, train=True):
        # 데이터셋 다운로드 및 로드
        if not os.path.isdir(root):
            os.makedirs(root)
            
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=download
        )
        self.train = train
        
        # Log-Mel Spectrogram Transform [cite: 1887]
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.sample_rate,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length,
            n_mels=Config.n_mels
        )
        
        # SpecAugment [cite: 1906]
        self.spec_augment = nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=35),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        
        # 길이 필터링: 320k 샘플 초과 시 None 반환 (collate_fn에서 처리) [cite: 1887]
        if waveform.shape[1] > Config.max_audio_length:
            return None
            
        # Log-Mel Spectrogram 변환
        # (Channel, n_mels, Time)
        spec = self.melspec(waveform)
        spec = torch.log(spec + 1e-9)
        
        if self.train:
            spec = self.spec_augment(spec)
        
        # (Channel, n_mels, Time) -> (n_mels, Time) -> Transpose -> (Time, n_mels)
        # DeepSpeech 모델 입력을 위해 (Channel, Freq, Time) 형태로 유지
        # 여기서는 반환 후 collate에서 배치 처리
        return spec.squeeze(0), transcript

def collate_fn(batch):
    # None 필터링 (길이 초과 데이터 제외)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None, None

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for spec, transcript in batch:
        # spec: (n_mels, Time)
        # Permute to (Time, n_mels) for pad_sequence
        spectrograms.append(spec.transpose(0, 1))
        
        # Label Encoding
        label = text_transform.text_to_int(transcript)
        labels.append(label)
        
        input_lengths.append(spec.shape[1]) # Time dimension
        label_lengths.append(len(label))

    # Padding
    # spectrograms: (Batch, Time, n_mels)
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1)
    # Restore to (Batch, Channel=1, n_mels, Time) for Conv2d
    spectrograms = spectrograms.permute(0, 1, 3, 2) 
    
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, torch.tensor(input_lengths), torch.tensor(label_lengths)

# ==========================================
# 4. DeepSpeech Model 
# ==========================================
class DeepSpeechAlgoPerf(nn.Module):
    def __init__(self, num_classes, input_dim=80, hidden_dim=512):
        super(DeepSpeechAlgoPerf, self).__init__()
        
        # 1. Convolution Subsampling [cite: 1907]
        # 입력 차원을 4배로 줄임. 보통 Time/2, Freq/2를 두 번 반복하거나 Stride 조절.
        # 여기서는 (2,2) Stride를 두 번 사용하여 Time과 Freq 모두 4배 축소
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), # 논문은 ReLU 명시 안했으나 Hardtanh/ReLU가 일반적
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        # RNN 입력 차원 계산
        # Freq축이 4배 줄어듦: input_dim // 4 * Channels(32)
        rnn_input_dim = 32 * (input_dim // 4)
        
        # 2. Recurrent Layers [cite: 1907]
        # 6 Bi-directional LSTM layers, internal dim 512
        self.rnn_layers = nn.ModuleList()
        self.rnn_bns = nn.ModuleList()
        self.num_rnn_layers = 6
        self.hidden_dim = hidden_dim
        
        for i in range(self.num_rnn_layers):
            self.rnn_layers.append(
                nn.LSTM(
                    input_size=rnn_input_dim if i == 0 else hidden_dim * 2,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    bias=True,
                    batch_first=True,
                    bidirectional=True
                )
            )
            # Batch Normalization inside LSTM layers as post normalization [cite: 1908]
            self.rnn_bns.append(nn.BatchNorm1d(hidden_dim * 2))

        # 3. Feed-forward Layers [cite: 1907]
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1), # Default dropout
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        # Xavier uniform initialization [cite: 1909]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)

    def forward(self, x, input_lengths):
        # x: (Batch, 1, n_mels, Time)
        
        # 1. Conv Subsampling
        x = self.conv(x) # (B, 32, F/4, T/4)
        
        # Prepare for RNN
        # Permute to (Batch, Time, Feature)
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T, C, F)
        x = x.view(B, T, -1) # (B, T, C*F)
        
        # Conv를 통과하며 줄어든 시퀀스 길이 업데이트
        output_lengths = input_lengths // 4
        
        # 2. Bi-LSTM with Residual & Post-BN
        for i in range(self.num_rnn_layers):
            residual = x
            
            out, _ = self.rnn_layers[i](x)
            
            # BN (requires B, C, T shape)
            out = out.permute(0, 2, 1) # (B, H, T)
            out = self.rnn_bns[i](out)
            out = out.permute(0, 2, 1) # (B, T, H)
            
            # Residual Connection (차원 일치 시 적용, 첫 레이어 제외)
            if i > 0:
                out += residual
            
            x = out

        # 3. Fully Connected
        x = self.fc(x) # (B, T, Classes)
        
        # CTC Loss를 위해 (Time, Batch, Classes) 및 LogSoftmax 적용
        x = F.log_softmax(x, dim=2)
        x = x.transpose(0, 1) # (T, B, C)
        
        return x, output_lengths

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    # 1. Data Setup
    # 실습을 위해 'train-clean-100' 사용. (다운로드 시간이 걸릴 수 있습니다)
    print("Loading LibriSpeech dataset...")
    train_dataset = AlgoPerfLibriSpeech(
        root=Config.data_path, 
        url="train-clean-100", 
        download=True, 
        train=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=True
    )

    # 2. Model Setup
    # Vocab size = characters + 1 (blank token for CTC)
    model = DeepSpeechAlgoPerf(num_classes=len(text_transform)).to(device)
    
    # 3. Loss & Optimizer
    # CTC Loss [cite: 1909]
    criterion = nn.CTCLoss(blank=len(text_transform)-1).to(device)
    # 논문은 AdamW, NadamW 등을 사용 [cite: 1722]
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    
    print("Start Training...")
    model.train()
    
    for epoch in range(Config.epochs):
        running_loss = 0.0
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_loader):
            if spectrograms is None: # 필터링된 배치 skip
                continue
                
            spectrograms = spectrograms.to(device) # (B, 1, F, T)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # outputs: (T, B, Classes), out_lengths: (B,)
            outputs, output_lengths = model(spectrograms, input_lengths)
            
            # Loss Calculation
            # CTC Loss requires input_lengths to be the length after subsampling
            loss = criterion(outputs, labels, output_lengths, label_lengths)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("Training Finished.")

if __name__ == "__main__":
    train()
