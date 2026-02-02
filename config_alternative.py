import os
import torch

# 1. Cihaz Seçimi (Varsa GPU kullanır)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Model İsmi (Hatanın sebebi burasıydı, bu satırı ekliyoruz)
# Türkçe için en iyi sonuç veren modellerden biridir.
# Alternatif: "intfloat/multilingual-e5-base" (Daha iyi anlamsal arama yapar)
HF_MODEL_ID = "/home/gunay/Masaüstü/Masaüstü/legal/project_root/output/MiniLM_weak_summary_to_reasoning" 

# 3. Veri Yolu (0 dosya sorununu çözmek için)
# Buraya terminalde gördüğün o doğru klasör yolunu açıkça yazıyoruz.
DATA_DIR = "/home/gunay/Masaüstü/Masaüstü/legal/seg_data"

# 4. Diğer Ayarlar
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4