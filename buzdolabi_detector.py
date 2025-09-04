import cv2
import numpy as np

def extract_refrigerator_region(image, segmentation_model):
    """
    Görselde buzdolabı bölgesini tespit eder ve kırpar
    
    Args:
        image: BGR formatında görsel
        segmentation_model: YOLOv8 segmentasyon modeli
        
    Returns:
        Buzdolabı bölgesi veya None
    """
    try:
        # Segmentasyon modelini çalıştır
        results = segmentation_model.predict(image, conf=0.5, verbose=False)
        
        if not results or not results[0].boxes:
            return None
            
        detection_result = results[0]
        
        # Buzdolabı sınıfını ara
        for i, class_id in enumerate(detection_result.boxes.cls):
            class_name = segmentation_model.names[int(class_id)]
            
            if class_name == "refrigerator":
                # Bounding box koordinatlarını al
                bbox = detection_result.boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Buzdolabı bölgesini kırp
                refrigerator_crop = image[y1:y2, x1:x2]
                
                return refrigerator_crop
                
        return None
        
    except Exception as e:
        print(f"Buzdolabı tespit hatası: {e}")
        return None
