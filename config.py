# config.py
import torch

# --- GENEL AYARLAR ---
SEED = 42
DATA_DIR = "/home/gunay/Masaüstü/legal/seg_data"       # JSON dosyalarının olduğu yer
OUT_DIR = "./output"      # Sonuçların ve modellerin kaydedileceği yer

# --- EĞİTİM AYARLARI ---
BATCH_SIZE = 4  #muhtemelen 8 yapmam gerekecek
EPOCHS = 3
LR = 2e-5
MAX_SEQ_LENGTH = 512      # <--- KRİTİK DÜZELTME (Eskiden 256 idi)
MAX_TRAIN = 2000          # Maksimum eğitim verisi sayısı
GRADIENT_ACCUMULATION_STEPS = 4

# --- DEĞERLENDİRME AYARLARI ---
EVAL_SIZE = 300           # <--- KRİTİK DÜZELTME (Test seti boyutu)
KS = (1, 5, 10)           # Hangi Recall değerlerine bakacağız?
ALPHA = 0.5               # Hybrid için BM25/Dense ağırlığı

# --- DONANIM ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
