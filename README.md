# Product Detection System

Bu proje, ticari buzdolaplarında ürün tespiti yapan bir YOLO tabanlı sistemdir.

## Özellikler

- **Buzdolabı Tespiti**: Görselde buzdolabı bölgesini otomatik tespit eder
- **Raf Segmentasyonu**: Beyaz rafları tespit ederek her rafı ayrı ayrı analiz eder
- **Ürün Tespiti**: Her rafta bulunan ürünleri tespit eder ve sayar
- **Web Arayüzü**: FastAPI tabanlı web uygulaması ile kolay kullanım

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Model dosyalarını indirin:
   - `yolov8n-seg.pt` (segmentasyon için)
   - `stajmodel_train6/weights/best.pt` (ürün tespiti için)

## Kullanım

1. Web uygulamasını başlatın:
```bash
python web_app.py
```

2. Tarayıcıda `http://127.0.0.1:8001` adresine gidin

3. Buzdolabı görselini yükleyin ve analiz edin

## Proje Yapısı

- `web_app.py` - FastAPI web uygulaması
- `analiz.py` - Ana analiz modülü
- `buzdolabi_detector.py` - Buzdolabı tespiti
- `shelf_detector.py` - Raf segmentasyonu
- `product_detector.py` - Ürün tespiti
- `model_config.py` - Model konfigürasyonu
- `examples/` - Örnek görseller ve analiz sonuçları

## Gereksinimler

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- FastAPI
- NumPy
- SciPy
