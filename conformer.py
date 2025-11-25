import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # Audio Params [cite: 1887]
    sample_rate = 16000
    n_fft = 400       # 25ms
    hop_length = 160  # 10ms
    n_mels = 80       # Log-mel spectrogram features
    
    # Model Params [cite: 1891-1892]
    encoder_dim = 512
    num_layers = 4
    num_heads = 8     # 512 dim / 64 per head = 8 heads (Typical default)
    depthwise_conv_kernel_size = 31 # Conformer default
    
    # Training Params
    batch_size = 8    # 메모리에 맞춰 조정
    learning_rate = 1e-3 # 논문의 NadamW/AdamW 사용 시 적절히 조정
    epochs = 1
    max_audio_length = 320000 # [cite: 1887]
    
    # Path
    data_path = "./data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {Config.device}")

# ==========================================
# 2. Tokenizer (Simple Character-level)
# ==========================================
class TextTransform:
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
# 3. Dataset & Preprocessing [cite: 1887-1889]
# ==========================================
class AlgoPerfLibriSpeech(Dataset):
    def __init__(self, root, url="train-clean-100", download=True, train=True):
        if not os.path.isdir(root):
            os.makedirs(root)
        
        # 실제 학습 시 train-clean-100, 360, other-500 조합 사용 [cite: 1881]
        # 여기서는 데모를 위해 train-clean-100만 사용
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=download
        )
        self.train = train
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.sample_rate,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length,
            n_mels=Config.n_mels
        )
        
        # SpecAugment is used in preprocessing pipeline
        self.spec_augment = nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=35),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=27)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        
        # Filter length > 320k [cite: 1887]
        if waveform.shape[1] > Config.max_audio_length:
            return None
            
        # Log-Mel Spectrogram [cite: 1887]
        spec = self.melspec(waveform)
        spec = torch.log(spec + 1e-9)
        
        if self.train:
            spec = self.spec_augment(spec)
        
        # (Channel, n_mels, Time) -> (Time, n_mels)
        return spec.squeeze(0).transpose(0, 1), transcript

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None, None

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for spec, transcript in batch:
        spectrograms.append(spec)
        
        label = text_transform.text_to_int(transcript)
        labels.append(label)
        
        input_lengths.append(spec.shape[0]) # Time dimension
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True) # (B, T, F)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, torch.tensor(input_lengths), torch.tensor(label_lengths)

# ==========================================
# 4. Conformer Model [cite: 1891-1892]
# ==========================================
class ConformerAlgoPerf(nn.Module):
    def __init__(self, num_classes, input_dim=80, encoder_dim=512, num_layers=4, num_heads=8):
        super(ConformerAlgoPerf, self).__init__()
        
        # torchaudio의 Conformer 구현체 사용
        self.conformer = torchaudio.models.Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=encoder_dim * 4, # Feed Forward dimension usually 4x encoder dim
            num_layers=num_layers,   # AlgoPerf: 4 layers [cite: 1892]
            depthwise_conv_kernel_size=Config.depthwise_conv_kernel_size
        )
        
        # Output Projection
        self.fc = nn.Linear(input_dim, num_classes) 
        # Note: torchaudio Conformer outputs same dim as input if input_dim match
        # But typically Conformer is (Input -> Linear -> Conformer -> Linear)
        # For simplicity matching dimension:
        self.input_projection = nn.Linear(input_dim, input_dim) 
        
        # Initialization [cite: 1892]
        self._init_weights()

    def _init_weights(self):
        # Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, input_lengths):
        # x: (Batch, Time, Freq)
        
        # Input Projection
        x = self.input_projection(x)
        
        # Conformer Forward
        # out: (Batch, Time, input_dim), lengths: (Batch,)
        out, out_lengths = self.conformer(x, input_lengths)
        
        # Final Linear
        out = self.fc(out) # (Batch, Time, Classes)
        
        # Log Softmax for CTC
        out = F.log_softmax(out, dim=2)
        
        # (Batch, Time, Classes) -> (Time, Batch, Classes)
        out = out.transpose(0, 1)
        
        return out, out_lengths

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    # 1. Data Setup
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
    # Vocab size + 1 (Blank)
    model = ConformerAlgoPerf(
        num_classes=len(text_transform),
        input_dim=Config.n_mels,
        encoder_dim=Config.encoder_dim,
        num_layers=Config.num_layers,
        num_heads=Config.num_heads
    ).to(Config.device)
    
    # 3. Loss & Optimizer
    # CTC Loss 
    criterion = nn.CTCLoss(blank=len(text_transform)-1).to(Config.device)
    
    # AlgoPerf uses AdamW/NadamW mostly
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    
    print(f"Conformer Model Created. Layers: {Config.num_layers}, Dim: {Config.encoder_dim}")
    print("Start Training...")
    
    model.train()
    for epoch in range(Config.epochs):
        running_loss = 0.0
        for batch_idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(train_loader):
            if spectrograms is None: continue
                
            spectrograms = spectrograms.to(Config.device) # (B, T, F)
            labels = labels.to(Config.device)
            input_lengths = input_lengths.to(Config.device)
            label_lengths = label_lengths.to(Config.device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs, output_lengths = model(spectrograms, input_lengths)
            
            # CTC Loss
            loss = criterion(outputs, labels, output_lengths, label_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN or Inf loss detected. Skipping batch.")
                continue

            loss.backward()
            
            # Gradient Clipping is often helpful for Conformer/Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("Training Finished.")

if __name__ == "__main__":
    train()
