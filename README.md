# Hukuki Asistan ⚖️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/Conference-ITTA%202026-blue)](#makale-ve-at%C4%B1f)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

**Derin Öğrenme Kullanarak Türk Hukukunda Gerekçeli Emsal Karar Tespiti ve Özetleme Sistemi**

Bu depo, taranmış (görüntü tabanlı) mahkeme kararları üzerinde bölüm farkındalıklı (section-aware) bilgi getirimi (information retrieval) yapmayı sağlayan **Hukuki Asistan** projesinin kaynak kodlarını içermektedir. Projemizin metodolojisi ve bulguları **ITTA 2026** konferansında sunulmak üzere hazırlanmıştır.

---

## 📖 Proje Hakkında

[cite_start]Türkiye'deki mahkeme kararları genellikle taranmış PDF'ler formatında sunulmaktadır[cite: 6, 23]. [cite_start]Bu durum, kararların içindeki yapısal metinlerin otomatik olarak ayrıştırılmasını zorlaştırmaktadır[cite: 6]. [cite_start]**Hukuki Asistan**, bir Görsel-Dil Modeli (VLM) kullanarak kararları "Olay", "Gerekçe" ve "Hüküm" bölümlerine ayırır[cite: 8, 28].

[cite_start]Özellikle **Gerekçe** (Reasoning) bölümünün, emsal kararların tespitinde en kritik retrieval sinyalini taşıdığı ampirik olarak doğrulanmıştır[cite: 11, 213].

## ✨ Temel Özellikler

* [cite_start]**VLM Tabanlı Ayıklama:** Taranmış PDF belgeleri Amazon Nova-2-Lite-V1 gibi modellerle yapılandırılmış JSON formatına dönüştürülür[cite: 85, 116].
* [cite_start]**Bölüm Farkındalıklı Bilgi Getirimi:** Sorgular tüm metin yerine belgelerin mantıksal bölümleri (Facts, Reasoning, Verdict) üzerinde çalıştırılır[cite: 8, 117].
* [cite_start]**Hibrit Arama Stratejisi:** Sözcük tabanlı (BM25) ve anlamsal (MPNet, XLM-R vb.) modellerin güçlerini birleştiren "Late Fusion" mimarisi kullanılır[cite: 9, 168].
* [cite_start]**Hukuki Mevzuat Desteği:** Kararlarda geçen kanun maddeleri (örn. TCK, TTK) otomatik olarak ayıklanır ve sözcük çapası (lexical anchor) olarak kullanılır[cite: 9, 118].

## 📊 Performans Bulguları

[cite_start]1.000 adet Sosyal Medya Hukuku kararı üzerinde yapılan testlerde[cite: 7, 79]:
* [cite_start]**En Güçlü Bölüm:** Tek başına aramalarda "Gerekçe" bölümü en yüksek başarıyı göstermiştir ($R@1=0.744$)[cite: 11, 213].
* [cite_start]**Hibrit Başarı:** MPNet ile yapılan bölüm farkındalıklı hibrit füzyon, $R@1=0.780$ ve $MRR@10=0.810$ skorlarına ulaşmıştır[cite: 11, 218].

## 📄 Makale ve Atıf

Çalışmamızı akademik projelerinizde kullanırsanız lütfen aşağıdaki şekilde atıfta bulununuz:

```bibtex
@inproceedings{kovaycin2026section,
  title={Section-Aware Retrieval for Turkish Case Law: A Case Study on Scanned Court Decisions},
  author={Kovay{\c{c}}in, K{\"u}bra and {\c{S}}i{\c{s}}er, Ece {\.I}rem and G{\"u}nay, Derda Sina and Ayka{\c{c}}, Yusuf Evren and Samet, Refik},
  booktitle={Proceedings of the ITTA 2026 Conference},
  year={2026},
  organization={ITTA}
}
