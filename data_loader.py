# data_loader.py
import os
import json
import glob
import numpy as np
from typing import List, Tuple, Dict
import config  # Ayarları çekiyoruz

def load_json_corpus(data_dir: str, pattern: str = "*.json", recursive: bool = False, limit: int = None):
    # Düzeltme: Senin dosyaların "vision_llm_processed_..." diye başlıyorsa pattern'i ona göre de ayarlayabiliriz
    # ama *.json hepsi için çalışır.
    search_path = os.path.join(data_dir, pattern)
    files = glob.glob(search_path, recursive=recursive)
    files.sort()
    
    if limit:
        files = files[:limit]
        
    raw_objs = []
    doc_ids = []
    
    print(f"Taranan klasör: {search_path}")
    print(f"Bulunan dosya sayısı: {len(files)}")
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Senin JSON yapındaki kilit anahtarlar var mı diye bakıyoruz
                if "rrl_segments" in data: 
                    raw_objs.append(data)
                    doc_ids.append(os.path.basename(fpath))
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
    return raw_objs, doc_ids

def extract_segments(obj: Dict) -> Dict[str, str]:
    """
    Senin özel JSON yapına göre parçaları ayıklar.
    """
    
    # 1. rrl_segments kutusunu açalım
    rrl = obj.get("rrl_segments", {})
    if rrl is None: rrl = {} 

    # --- KRİTİK DÜZELTME BURADA ---
    # Senin JSON'da isimler 'facts_text', 'reasoning_text' olduğu için aynısını yazıyoruz.
    facts = rrl.get("facts_text", "")
    reasoning = rrl.get("reasoning_text", "")
    verdict = rrl.get("verdict_text", "") # Hüküm kısmı (lazım olursa diye)

    # 2. Kanun Maddeleri (structural_features içinde liste olarak duruyor)
    struct = obj.get("structural_features", {})
    if struct is None: struct = {}
    
    laws_list = struct.get("mentioned_laws", [])
    # Liste halindeyse metne çevirelim (Örn: ["IK 72"] -> "IK 72")
    if isinstance(laws_list, list):
        laws = ", ".join(str(x) for x in laws_list)
    else:
        laws = str(laws_list)

    # 3. Özet bilgisi
    summary = obj.get("summary_for_model", "")
    
    # 4. Tam metin oluşturma
    # Senin JSON'da 'full_text' yoktu, biz parçaları birleştirip yapıyoruz.
    full = f"OLAYLAR: {facts}\nKANUNLAR: {laws}\nGEREKÇE: {reasoning}\nHÜKÜM: {verdict}".strip()
    
    return {
        "facts": facts,
        "laws": laws,
        "facts_laws": f"{facts}\n{laws}".strip(),
        "reasoning": reasoning,
        "full": full,
        "summary": summary
    }

def build_pool(raw_objs, doc_ids, variant="summary_to_reasoning"):
    q_all = []
    d_all = []
    ids_all = []
    miss_q = 0
    miss_d = 0
    
    for idx, obj in enumerate(raw_objs):
        segs = extract_segments(obj)
        q_text = segs["summary"]
        
        # Senin istediğin varyant (Özet -> Gerekçe Eşleşmesi)
        if variant == "summary_to_reasoning":
            d_text = segs["reasoning"]
        else:
            d_text = segs["full"]
            
        # Boş veri kontrolü
        if not q_text or not q_text.strip():
            miss_q += 1
            continue
        if not d_text or not d_text.strip():
            miss_d += 1
            continue
            
        q_all.append(q_text)
        d_all.append(d_text)
        ids_all.append(doc_ids[idx])
        
    return q_all, d_all, ids_all, miss_q, miss_d

def disjoint_split(n: int, eval_size: int, max_train: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    # Test setini belirle
    eval_size_eff = min(eval_size, n - 1) if n > 1 else 0
    
    if eval_size_eff < 1:
        eval_size_eff = 0

    eval_idxs = idx[:eval_size_eff].tolist()
    remaining = idx[eval_size_eff:]
    
    train_cap = min(max_train, len(remaining))
    train_idxs = remaining[:train_cap].tolist()

    return train_idxs, eval_idxs