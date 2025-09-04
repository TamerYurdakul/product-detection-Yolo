from ultralytics import YOLO
import cv2
import numpy as np

def calculate_iou(box1, box2):
    """
    İki kutu arasında IoU (Intersection over Union) hesaplar
    
    Args:
        box1, box2: (x1, y1, x2, y2) formatında kutular
        
    Returns:
        float: IoU değeri (0-1 arası)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Kesişim alanını hesapla
    x1_intersect = max(x1_1, x1_2)
    y1_intersect = max(y1_1, y1_2)
    x2_intersect = min(x2_1, x2_2)
    y2_intersect = min(y2_1, y2_2)
    
    # Kesişim yoksa IoU = 0
    if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
        return 0.0
    
    # Kesişim alanı
    intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
    
    # Her kutunun alanı
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Birleşim alanı
    union_area = area1 + area2 - intersection_area
    
    # IoU hesapla
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def custom_nms(detections, iou_threshold=0.2, cross_class_iou_threshold=0.3):
    """
    Gelişmiş Non-Maximum Suppression uygular
    - Aynı sınıflar için normal NMS
    - Farklı sınıflar arası da IoU kontrolü (duplicate önleme)
    
    Args:
        detections: [(x1, y1, x2, y2, class_name, confidence), ...] formatında liste
        iou_threshold: Aynı sınıf için IoU eşik değeri
        cross_class_iou_threshold: Farklı sınıflar arası IoU eşik değeri
        
    Returns:
        list: Filtrelenmiş detections listesi
    """
    if not detections:
        return []
    
    # Confidence'a göre sırala (yüksekten düşüğe)
    detections = sorted(detections, key=lambda x: x[5], reverse=True)
    
    filtered_detections = []
    
    while detections:
        # En yüksek confidence'lı detection'ı al
        current_detection = detections.pop(0)
        current_class = current_detection[4]
        current_box = current_detection[:4]
        current_confidence = current_detection[5]
        
        # Bu detection'ı kabul edip etmeyeceğimizi kontrol et
        should_keep = True
        
        # Daha önce kabul edilen detection'larla çakışma kontrolü
        for accepted_detection in filtered_detections:
            accepted_box = accepted_detection[:4]
            accepted_class = accepted_detection[4]
            accepted_confidence = accepted_detection[5]
            
            iou = calculate_iou(current_box, accepted_box)
            
            # Debug: IoU hesaplama sonucunu logla
            if iou > 0.1:  # Sadece anlamlı IoU değerlerini logla
                print(f"🔍 IoU hesaplama: {current_class} vs {accepted_class} = {iou:.3f}")
            
            # Aynı sınıftan ise daha düşük threshold kullan
            if accepted_class == current_class:
                if iou > iou_threshold:
                    should_keep = False
                    print(f"❌ Aynı sınıf NMS: {current_class} (IoU: {iou:.2f} > {iou_threshold})")
                    break
            else:
                # Özel durum: Kızılay ürünleri için daha agresif filtreleme
                is_kizilay_duplicate = (
                    ("kizil" in current_class.lower() and "kizil" in accepted_class.lower()) or
                    ("kizilay" in current_class.lower() and "kizilay" in accepted_class.lower()) or
                    ("kizil" in current_class.lower() and "kizilay" in accepted_class.lower()) or
                    ("kizilay" in current_class.lower() and "kizil" in accepted_class.lower())
                )
                
                # Kızılay ürünleri için özel threshold
                effective_threshold = cross_class_iou_threshold
                if is_kizilay_duplicate:
                    effective_threshold = 0.15  # ÇOK ÇOK düşük threshold - süper agresif filtreleme
                    print(f"🔍 Kızılay duplicate kontrolü: {current_class} vs {accepted_class} (IoU: {iou:.2f}, threshold: {effective_threshold})")
                
                # Farklı sınıftan ama yüksek IoU varsa (aynı fiziksel obje olabilir)
                if iou > effective_threshold:
                    # Confidence'ı daha yüksek olanı tercih et
                    if current_confidence <= accepted_confidence:
                        should_keep = False
                        print(f"❌ Cross-class NMS: {current_class} vs {accepted_class} (IoU: {iou:.2f}, conf: {current_confidence:.2f} <= {accepted_confidence:.2f})")
                        break
                    else:
                        # Mevcut detection daha yüksek confidence'a sahip, eskisini çıkar
                        filtered_detections.remove(accepted_detection)
                        print(f"✅ Cross-class replacement: {current_class} replaced {accepted_class} (IoU: {iou:.2f}, conf: {current_confidence:.2f} > {accepted_confidence:.2f})")
        
        if should_keep:
            filtered_detections.append(current_detection)
        
        # Kalan detection'ları kontrol et
        remaining_detections = []
        for detection in detections:
            detection_box = detection[:4]
            detection_class = detection[4]
            detection_confidence = detection[5]
            
            # Current detection ile IoU kontrol et
            iou = calculate_iou(current_box, detection_box)
            
            # Aynı sınıftan ise
            if detection_class == current_class and should_keep:
                if iou < iou_threshold:
                    remaining_detections.append(detection)
                else:
                    print(f"❌ Remaining same-class filtered: {detection_class} (IoU: {iou:.2f})")
            # Farklı sınıftan ise
            elif detection_class != current_class:
                # Kızılay duplicate kontrolü
                is_kizilay_duplicate = (
                    ("kizil" in current_class.lower() and "kizil" in detection_class.lower()) or
                    ("kizilay" in current_class.lower() and "kizilay" in detection_class.lower()) or
                    ("kizil" in current_class.lower() and "kizilay" in detection_class.lower()) or
                    ("kizilay" in current_class.lower() and "kizil" in detection_class.lower())
                )
                
                # Threshold belirleme
                effective_threshold = cross_class_iou_threshold
                if is_kizilay_duplicate:
                    effective_threshold = 0.15  # Süper agresif
                
                if should_keep and iou > effective_threshold:
                    # Confidence karşılaştır
                    if detection_confidence <= current_confidence:
                        print(f"❌ Remaining cross-class filtered: {detection_class} (IoU: {iou:.2f}, conf: {detection_confidence:.2f})")
                    else:
                        remaining_detections.append(detection)
                else:
                    remaining_detections.append(detection)
            else:
                remaining_detections.append(detection)
        
        detections = remaining_detections
    
    return filtered_detections

def detect_products_in_shelf(shelf_image, model_path):
    """
    Raf görselindeki ürünleri tespit eder
    
    Args:
        shelf_image: BGR formatında raf görseli
        model_path: YOLO model dosya yolu
        
    Returns:
        tuple: (ürün_sayıları, toplam_ürün, bilinmeyen_kutular, bilinen_kutular)
    """
    try:
        # Her raf için yeni model yükle (daha doğru sonuç için)
        product_model = YOLO(model_path)
        
        # Ürün tespiti yap - optimize edilmiş parametreler
        detection_results = product_model.predict(
            shelf_image, 
            conf=0.6,      # Normal confidence
            iou=0.5,       # Normal IoU threshold
            verbose=False
        )
        
        if not detection_results:
            return {}, 0, [], []
            
        # Tüm model sınıflarını al
        all_classes = list(product_model.names.values())
        
        # İlk geçiş: Tüm detection'ları topla
        raw_detections = []
        unknown_boxes = []
        
        for result in detection_results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                # Tespit bilgilerini al
                class_id = int(box.cls[0])
                product_name = result.names.get(class_id, "bilinmeyen")
                confidence_score = float(box.conf[0])
                
                # Bounding box koordinatları
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Geçersiz kutu kontrolü
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Çok küçük kutuları filtrele (min 30x30 pixel - daha agresif)
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width < 30 or box_height < 30:
                    print(f"❌ Çok küçük kutu filtrelendi: {box_width}x{box_height}")
                    continue
                
                # Çok büyük kutuları da filtrele (muhtemelen hatalı tespit)
                shelf_height, shelf_width = shelf_image.shape[:2]
                if (box_width > shelf_width * 0.8) or (box_height > shelf_height * 0.8):
                    print(f"❌ Çok büyük kutu filtrelendi: {box_width}x{box_height} (raf: {shelf_width}x{shelf_height})")
                    continue
                
                # Bilinmeyen ürün kontrolü
                if product_name not in all_classes:
                    product_name = "bilinmeyen"
                    unknown_boxes.append((x1, y1, x2, y2))
                    continue
                
                # Ek confidence filtresi (ürün tipine göre) - Debug için düşük
                min_confidence = 0.5   # Genel minimum - debug için düşük
                if "kizil" in product_name.lower():
                    min_confidence = 0.45  # Kızılay ürünleri için daha düşük
                elif "dimes" in product_name.lower():
                    min_confidence = 0.55  # Dimes için biraz daha yüksek
                
                if confidence_score < min_confidence:
                    print(f"❌ Düşük confidence filtrelendi: {product_name} ({confidence_score:.2f} < {min_confidence})")
                    continue
                
                # Ham detection'ı listeye ekle
                raw_detections.append((x1, y1, x2, y2, product_name, confidence_score))

        
        # Custom NMS uygula
        filtered_detections = custom_nms(raw_detections, iou_threshold=0.5)
        
        # İkinci geçiş: Filtrelenmiş detection'ları işle - EXTRA DUPLICATE KONTROL
        product_counts = {}
        total_product_count = 0
        known_boxes = []
        

        
        for detection in filtered_detections:
            x1, y1, x2, y2, product_name, confidence_score = detection
            
            # EXTRA KONTROL: known_boxes'ta aynı pozisyonda kutu var mı?
            is_final_duplicate = False
            for existing_box in known_boxes:
                ex1, ey1, ex2, ey2, ex_name, ex_conf = existing_box
                
                # Koordinat benzerliği kontrolü (5 pixel tolerans)
                if (abs(x1 - ex1) < 5 and abs(y1 - ey1) < 5 and 
                    abs(x2 - ex2) < 5 and abs(y2 - ey2) < 5):
                    is_final_duplicate = True

                    break
            
            if not is_final_duplicate:
                # Ürün sayısını güncelle
                if product_name not in product_counts:
                    product_counts[product_name] = {'count': 0}
                
                # Altılı paket özel sayımı
                if "altili" in product_name.lower():
                    product_counts[product_name]['count'] += 1
                    total_product_count += 6  # Altılı paket = 6 ürün
                else:
                    product_counts[product_name]['count'] += 1
                    total_product_count += 1
                
                # Bilinen kutular listesine ekle
                known_boxes.append((x1, y1, x2, y2, product_name, confidence_score))

        
        return product_counts, total_product_count, unknown_boxes, known_boxes
        
    except Exception as e:
        print(f"Ürün tespit hatası: {e}")
        return {}, 0, [], []
