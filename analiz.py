import cv2
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Any
import os

# Kendi modüllerimizi import et
from product_detector import detect_products_in_shelf
from shelf_detector import create_shelf_mask
from buzdolabi_detector import extract_refrigerator_region
from model_config import get_segmentation_model, get_detection_model, DETECTION_MODEL
# Product dimensions removed - not needed
from ultralytics import YOLO

def analyze_full_image(image):
    """
    Buzdolabı tespit edilemediğinde tüm görsel üzerinde ürün tespiti yapar
    """
    try:
        # Tüm görsel üzerinde ürün tespiti yap
        product_counts, total_product_count, unknown_boxes, known_boxes = detect_products_in_shelf(
            image, DETECTION_MODEL
        )
        
        # Sonuçları düzenle
        shelf_products = {}
        for product_name, product_info in product_counts.items():
            if isinstance(product_info, dict) and 'count' in product_info:
                shelf_products[product_name] = product_info['count']
            else:
                shelf_products[product_name] = product_info
        
        # Ürünleri görsel üzerine çiz
        image_with_boxes = draw_product_boxes(image.copy(), known_boxes, unknown_boxes)
        
        # BGR'den RGB'ye çevir (web görünümü için)
        final_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        
        # Tek raf olarak sonuç döndür
        shelf_results = [{
            "raf_no": 1,
            "urunler": shelf_products,
            "toplam": sum(shelf_products.values()) if shelf_products else 0
        }]
        
        return {
            "toplam_urun": sum(shelf_products.values()) if shelf_products else 0,
            "raf_bilgileri": shelf_results,
            "gorsel": final_image,  # Web uygulaması bu ismi arıyor
            # Alan/kaplama hesapları ve ham kutular arayüzde kullanılmıyor
        }
        
    except Exception as e:
        print(f"Tam görsel analiz hatası: {e}")
        return {"error": f"Görsel analiz hatası: {str(e)}"}

def raf_analizi_yap(image, enhance: bool = False, use_ensemble: bool = False) -> Dict[str, Any]:
    """
    Buzdolabı görselini analiz ederek raf bazlı ürün tespiti yapar
    
    Args:
        image: RGB veya BGR formatında görsel
        enhance: Kontrast iyileştirme (kullanılmıyor)
        use_ensemble: Ensemble tahmin (kullanılmıyor)
        
    Returns:
        Dict: Analiz sonuçları
    """
    try:
        # RGB formatından BGR'ye çevir (OpenCV için)
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            processed_image = image.copy()
        
        # 1. Buzdolabı bölgesini tespit et ve kırp
        segmentation_model = get_segmentation_model()
        refrigerator_crop = extract_refrigerator_region(processed_image, segmentation_model)
        
        # Eğer buzdolabı tespit edilemezse, tüm görseli kullan
        if refrigerator_crop is None:
            print("⚠️ Buzdolabı tespit edilemedi, tüm görsel analiz ediliyor...")
            return analyze_full_image(processed_image)

        # 3. Beyaz rafları tespit et
        shelf_mask = create_shelf_mask(refrigerator_crop)
        
        # Dikey projeksiyon ile raf sınırlarını bul - ULTRA SIKI PARAMETRELER
        vertical_projection = np.sum(shelf_mask, axis=1)
        shelf_boundaries, _ = find_peaks(
            vertical_projection, 
            distance=250,      # Raflar arası minimum mesafe maksimum
            prominence=25000,  # Prominence ultra yüksek (sadece ana raflar)
            height=30000       # Minimum yükseklik ultra yüksek
        )
        
        # Üst/alt sınırları da ekle (üst rafı kaçırmamak için)
        height = refrigerator_crop.shape[0]
        shelf_boundaries = np.array([0] + list(shelf_boundaries) + [height])
        shelf_boundaries = np.unique(shelf_boundaries)
        shelf_boundaries = np.sort(shelf_boundaries)

        print(f"✅ {len(shelf_boundaries)} raf sınırı bulundu")
        
        if len(shelf_boundaries) < 2:
            print("❌ Yeterli raf sınırı bulunamadı")
            return {"error": "Raf sınırları tespit edilemedi"}

        # 4. Her rafı ayrı ayrı analiz et (AYNI MODEL İLE)
        total_products = 0
        shelf_results = []
        


        for shelf_index in range(len(shelf_boundaries) - 1):

            
            # Raf görselini kırp
            shelf_start = shelf_boundaries[shelf_index]
            shelf_end = shelf_boundaries[shelf_index + 1]
            shelf_image = refrigerator_crop[shelf_start:shelf_end, :]
            
            # Bu raftaki ürünleri tespit et
            product_counts, shelf_total, unknown_boxes, known_boxes = detect_products_in_shelf(
                shelf_image, DETECTION_MODEL
            )
            
            total_products += shelf_total
            
            # Bu raf için ürün listesi oluştur
            shelf_products = {}
            for product_name, info in product_counts.items():
                shelf_products[product_name] = info["count"]
            
            # Alan kaplama hesaplaması kaldırıldı
            
            

            # 4. Tespit edilen ürünleri görsel üzerine çiz
            shelf_image_with_boxes = draw_product_boxes(shelf_image.copy(), known_boxes, unknown_boxes)
            
            # Çizilmiş raf görselini ana görsele geri yerleştir
            refrigerator_crop[shelf_start:shelf_end, :] = shelf_image_with_boxes
            


            # Raf sonuçlarını kaydet (kaplama hesapları kaldırıldı)
            shelf_results.append({
                "raf_no": shelf_index + 1,
                "urunler": shelf_products,
                "bilinmeyen_kutular": unknown_boxes
            })

        # BGR'den RGB'ye çevir (web görünümü için)
        final_image = cv2.cvtColor(refrigerator_crop, cv2.COLOR_BGR2RGB)



        # Sonuçları döndür
        return {
            "toplam_urun": total_products,
            "raf_bilgileri": shelf_results,
            "gorsel": final_image,  # Web uygulaması bu ismi arıyor
            "kaplama_yuzdesi": 0.0,  # Web uyumluluğu için
            "boxes_xyxy": [],        # Web uyumluluğu için
            "classes": [],           # Web uyumluluğu için
            "scores": []             # Web uyumluluğu için
        }

    except Exception as e:
        print(f"❌ Analiz hatası: {e}")
        return {"error": f"Raf analizi hatası: {str(e)}"}

def draw_product_boxes(shelf_image, known_boxes, unknown_boxes):
    """
    Raf görselinin üzerine ürün kutularını çizer - DUPLICATE ETİKET ÖNLEYİCİ
    
    Args:
        shelf_image: Raf görseli
        known_boxes: Bilinen ürün kutuları
        unknown_boxes: Bilinmeyen ürün kutuları
        
    Returns:
        Kutuları çizilmiş görsel
    """
    try:
        # Çizilen etiketlerin pozisyonlarını takip et
        drawn_positions = []
        
        # Bilinen ürünler için yeşil kutular
        for item in known_boxes:
            if len(item) == 6:  # Format: (x1, y1, x2, y2, cls_name, confidence)
                x1, y1, x2, y2, product_name, confidence = item
                label = f"{product_name}: {confidence:.2f}"
            else:  # Eski format: (x1, y1, x2, y2, cls_name)
                x1, y1, x2, y2, product_name = item
                label = product_name
            
            # Bu pozisyonda daha önce etiket çizilmiş mi kontrol et
            current_pos = (x1, y1, x2, y2)
            is_duplicate = False
            
            for drawn_pos in drawn_positions:
                dx1, dy1, dx2, dy2 = drawn_pos
                # Eğer kutular çok yakınsa (10 pixel tolerans)
                if (abs(x1 - dx1) < 10 and abs(y1 - dy1) < 10 and 
                    abs(x2 - dx2) < 10 and abs(y2 - dy2) < 10):
                    is_duplicate = True
                    print(f"🚫 DUPLICATE ETİKET ENGELLENDI: {label} at ({x1},{y1})-({x2},{y2})")
                    break
            
            if not is_duplicate:
                # Yeşil dikdörtgen çiz
                cv2.rectangle(shelf_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Etiket için arka plan ve metin
                draw_label_with_background(shelf_image, label, (x1, y1), (0, 255, 0))
                
                # Bu pozisyonu kaydet
                drawn_positions.append(current_pos)
                print(f"✅ ETİKET ÇİZİLDİ: {label} at ({x1},{y1})-({x2},{y2})")

        # Bilinmeyen ürünler için kırmızı kutular
        for box_coords in unknown_boxes:
            x1, y1, x2, y2 = box_coords
            
            # Kırmızı dikdörtgen çiz
            cv2.rectangle(shelf_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Bilinmeyen etiketi
            draw_label_with_background(shelf_image, "Bilinmeyen", (x1, y1), (0, 0, 255))

        return shelf_image
        
    except Exception as e:
        print(f"❌ Kutu çizim hatası: {e}")
        return shelf_image

def draw_label_with_background(image, text, position, color):
    """
    Arka planlı etiket çizer
    
    Args:
        image: Görsel
        text: Yazılacak metin
        position: (x, y) pozisyonu
        color: BGR renk
    """
    try:
        x, y = position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Metin boyutunu hesapla
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Etiket pozisyonunu ayarla (kutu üstünde)
        text_y = max(text_height + 10, y - 10)
        bg_y1 = text_y - text_height - 8
        bg_y2 = text_y + 8
        
        # Sınır kontrolü
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(image.shape[1], x + text_width + 20)
        
        # Arka plan dikdörtgeni çiz
        cv2.rectangle(image, (x, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # İnce siyah çerçeve ekle
        cv2.rectangle(image, (x, bg_y1), (bg_x2, bg_y2), (0, 0, 0), 1)
        
        # Beyaz metni yaz
        cv2.putText(image, text, (x + 10, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        print(f"🏷️ Etiket çizildi: '{text}' pozisyon ({x}, {text_y})")
        
    except Exception as e:
        print(f"❌ Etiket çizim hatası: {e}")
        import traceback
        traceback.print_exc()

# Test fonksiyonu
if __name__ == "__main__":
    print("🚀 Raf Analizi Test Modülü")
    print("Web uygulaması üzerinden test edin!")
    
    # Basit model kontrolü
    try:
        from model_config import DETECTION_MODEL
        if os.path.exists(DETECTION_MODEL):
            print("✅ Ana model dosyası mevcut")
        else:
            print("❌ Model dosyası bulunamadı:", DETECTION_MODEL)
    except Exception as e:
        print("❌ Model kontrolü hatası:", e)