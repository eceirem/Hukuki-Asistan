import numpy as np
import torch
import config
from data_loader import load_json_corpus, build_pool
import models
import evaluation

def main():
    # ---------------------------------------------------------
    # 1. ADIM: TÜM VERİ SETİNİ YÜKLE (Kütüphaneyi Oluştur)
    # ---------------------------------------------------------
    print("📚 Veri seti yükleniyor...")
    
    # Burada 'limit=None' diyerek tüm veriyi çekiyoruz, split yapmıyoruz.
    raw_objs, doc_ids = load_json_corpus(
        data_dir=config.DATA_DIR, # Config dosyasında tanımlı olmalı
        pattern="*.json",         # Senin dosya isimlendirmene göre ayarla
        recursive=False
    )

    # Arama Havuzu (Corpus) oluşturuluyor
    # variant="reasoning" veya "full" seçebilirsin, neyin içinde arayacaksan.
    _, d_texts, ids_all, _, _ = build_pool(raw_objs, doc_ids, variant="summary_to_reasoning")
    
    print(f"✅ Toplam {len(d_texts)} adet doküman indekslenmeye hazır.")

    # ---------------------------------------------------------
    # 2. ADIM: MODELLERİ HAZIRLA VE İNDEKSLE
    # ---------------------------------------------------------
    print("🧠 Modeller yükleniyor ve indeks oluşturuluyor (biraz sürebilir)...")
    
    # A) Dense Model (Vektör)
    model = models.make_st_from_hf(config.HF_MODEL_ID) # Config'den model ismini çeker
    
    # Tüm dokümanların vektörlerini (embedding) bir kere hesapla
    d_embs = models.st_encode(model, d_texts)
    
    # B) Sparse Model (BM25)
    bm25 = evaluation.build_bm25_index(d_texts)
    
    print("🚀 Sistem hazır! Sorgu bekleniyor...")

    # ---------------------------------------------------------
    # 3. ADIM: CANLI ARAMA DÖNGÜSÜ
    # ---------------------------------------------------------
    while True:
        print("\n" + "="*50)
        query_text = input("🔍 Sorgunuzu girin (Çıkış için 'q'): ").strip()
        
        if query_text.lower() == 'q':
            print("Çıkış yapılıyor...")
            break
        
        if not query_text:
            continue

        # --- ARAMA İŞLEMİ ---
        
        # 1. Vektör Araması Skoru
        q_emb = models.st_encode(model, [query_text]) # (1, 768)
        dense_scores = models.dense_score_matrix(q_emb, d_embs) # (1, N)
        
        # 2. BM25 Araması Skoru
        bm25_raw = evaluation.bm25_scores(bm25, query_text) # (N,)
        # Boyut uyuşmazlığı olmaması için (1, N) formatına getiriyoruz
        bm25_scores = bm25_raw.reshape(1, -1)
        
        # 3. Hibrit Birleştirme (Alpha ayarı config'den veya elle)
        # alpha=1.0 sadece BM25, alpha=0.0 sadece Vektör. 0.5 ikisinin ortası.
        final_scores = evaluation.hybrid_scores(bm25_scores, dense_scores, alpha=0.5)
        
        # Skoru tek boyuta indir (N,)
        final_scores = final_scores.flatten() 
        
        # --- SONUÇLARI SIRALA VE GÖSTER ---
        
        top_k = 15
        # En yüksek skordan en düşüğe sırala (büyükten küçüğe olduğu için - ile çarpıp argsort)
        top_indices = np.argsort(-final_scores)[:top_k]
        
        print(f"\n🏆 En Benzer {top_k} Sonuç:\n")
        
        for rank, idx in enumerate(top_indices):
            doc_id = ids_all[idx]
            score = final_scores[idx]
            content = d_texts[idx]
            
            # İçeriğin çok uzunsa sadece başını gösterelim
            preview = content[:300] + "..." if len(content) > 300 else content
            
            print(f"{rank+1}. [Skor: {score:.4f}] Dosya: {doc_id}")
            print(f"   İçerik: {preview}")
            print("-" * 30)

if __name__ == "__main__":
    main()