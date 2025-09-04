import os
from ultralytics import YOLO

# Model dosya yolları
SEGMENTATION_MODEL = "yolov8n-seg.pt"
DETECTION_MODEL = os.path.join("stajmodel_train6", "weights", "best.pt")

# Model yükleme fonksiyonları
def get_segmentation_model():
    """Buzdolabı segmentasyon modelini yükler"""
    return YOLO(SEGMENTATION_MODEL)

def get_detection_model():
    """Ürün tespit modelini yükler"""
    return YOLO(DETECTION_MODEL)

# Ürün boyutları ve alan hesapları kaldırıldı
