# main.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Modüllerimiz
import config
import data_loader
import models
import trainer
import evaluation

def run_experiment(name, model_obj, train_data, eval_data, bm25_pool_mat):
    """
    Tek bir deney döngüsü: Eğit -> Test Et -> Raporla
    """
    print(f"\n[{name}] Deney Başlatılıyor...")
    
    t_queries, t_docs = train_data
    e_queries, e_true_idx, d_reasoning_all = eval_data
    
    # 1. EĞİTİM
    out_dir = os.path.join(config.OUT_DIR, f"{name}_weak_summary_to_reasoning")
    model = trainer.train_weak_triplet(model_obj, t_queries, t_docs, out_dir)
    
    # 2. VEKTÖRLEŞTİRME (Evaluation)
    # Tüm havuzu (1000 doküman) ve sorguları vektöre çevir
    print(f"[{name}] Embedding calculation...")
    d_embs = models.st_encode(model, d_reasoning_all)
    q_embs = models.st_encode(model, e_queries)
    
    # 3. DENSE SKORLAMA
    dense_mat = models.dense_score_matrix(q_embs, d_embs)
    dense_metrics = evaluation.recall_at_k_from_scores(dense_mat, e_true_idx, ks=config.KS)
    
    # 4. HYBRID SKORLAMA
    # Önceden hesaplanmış BM25 matrisi ile Dense matrisini birleştir
    hyb_mat = evaluation.hybrid_scores(bm25_pool_mat, dense_mat, config.ALPHA)
    hyb_metrics = evaluation.recall_at_k_from_scores(hyb_mat, e_true_idx, ks=config.KS)
    
    print(f"[{name}] SONUÇ -> Dense R@10: {dense_metrics['R@10']:.3f} | Hybrid R@10: {hyb_metrics['R@10']:.3f}")
    
    return dense_metrics, hyb_metrics, out_dir

def main():
    # --- 1. VERİ YÜKLEME ---
    print(f"Veriler yükleniyor: {config.DATA_DIR}")
    # Önce ana klasöre, bulamazsa alt klasörlere bak
    raw_objs, doc_ids = data_loader.load_json_corpus(config.DATA_DIR, "*.json", recursive=False)
    if not raw_objs:
        raw_objs, doc_ids = data_loader.load_json_corpus(config.DATA_DIR, "**/*.json", recursive=True)
        
    print(f"Yüklenen Doküman Sayısı: {len(raw_objs)}")
    assert len(raw_objs) > 0, "HATA: Hiç JSON dosyası bulunamadı!"

    # --- 2. VERİ HAZIRLAMA (POOLING) ---
    # Varyant: Summary -> Reasoning
    q_all, d_reasoning_all, ids_all, missq, missd = data_loader.build_pool(raw_objs, doc_ids, variant="summary_to_reasoning")
    print(f"Veri Hazır: {len(ids_all)} adet. (Eksik Sorgu: {missq}, Eksik Gerekçe: {missd})")
    
    # --- 3. SPLIT (EĞİTİM / TEST AYRIMI) ---
    n = len(ids_all)
    train_idxs, eval_idxs = data_loader.disjoint_split(n, config.EVAL_SIZE, config.MAX_TRAIN, seed=config.SEED)
    
    # İndeksleri kullanarak verileri ayır
    train_queries = [q_all[i] for i in train_idxs]
    train_docs    = [d_reasoning_all[i] for i in train_idxs]
    
    eval_queries  = [q_all[i] for i in eval_idxs]
    eval_true_idx = eval_idxs # Havuz içindeki doğru indeksler
    
    print(f"Eğitim Seti: {len(train_queries)} | Test Seti: {len(eval_queries)}")

    # --- 4. GLOBAL BM25 (HYBRID İÇİN) ---
    print("\nHybrid Fusion için BM25 skorları önden hesaplanıyor...")
    # BM25 için Facts+Laws kısımlarını çekiyoruz
    bm25_docs_full = []
    for obj in raw_objs:
        bm25_docs_full.append(data_loader.extract_segments(obj)["facts_laws"])
    
    # Hizalama (Alignment)
    id_to_corpus = {did: i for i, did in enumerate(doc_ids)}
    pool_corpus_idxs = [id_to_corpus[did] for did in ids_all]
    bm25_pool_docs = [bm25_docs_full[i] for i in pool_corpus_idxs]
    
    # BM25 İndeksle ve Skorla
    bm25_obj = evaluation.build_bm25_index(bm25_pool_docs)
    bm25_pool_mat = np.stack([evaluation.bm25_scores(bm25_obj, q) for q in eval_queries], axis=0)
    
    # BM25 Baseline Skorunu Yazdır (Makale Tablo 5 için)
    bm25_metrics = evaluation.recall_at_k_from_scores(bm25_pool_mat, eval_true_idx, ks=config.KS)
    print(f"Baseline BM25 (Facts+Laws) R@10: {bm25_metrics['R@10']:.4f}")

    # --- 5. MODELLER VE DENEYLER ---
    # (Model Adı, Oluşturucu Fonksiyon)
    experiment_specs = [
        # Hafif Modeller
        ("MiniLM", lambda: SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=config.DEVICE)),
        # Yerli Gururumuz (512 Token Ayarlı)
        ("BERTurk-Legal", lambda: models.make_st_from_hf("KocLab-Bilkent/BERTurk-Legal", max_seq_length=config.MAX_SEQ_LENGTH, pooling="mean")),
        # Global Dev (512 Token Ayarlı)
        ("XLM-R (Base)", lambda: models.make_st_from_hf("xlm-roberta-base", max_seq_length=config.MAX_SEQ_LENGTH, pooling="mean")),
    ]
    
    results = []
    train_pack = (train_queries, train_docs)
    eval_pack = (eval_queries, eval_true_idx, d_reasoning_all)
    
    for name, builder in experiment_specs:
        model = builder()
        d_met, h_met, path = run_experiment(name, model, train_pack, eval_pack, bm25_pool_mat)
        results.append((name, d_met, h_met, path))
        
    # --- 6. FİNAL RAPORU ---
    print("\n================ FINAL SUMMARY ================")
    print(f"Baseline BM25 : {bm25_metrics}")
    for name, dmet, hmet, path in results:
        print(
            f"{name:>15} | Dense R@10={dmet['R@10']:.3f} "
            f"| Hybrid R@10={hmet['R@10']:.3f} "
        )

if __name__ == "__main__":
    main()
