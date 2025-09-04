import numpy as np
import cv2

def create_shelf_mask(image):
    """
    Buzdolabındaki beyaz rafları tespit eden maske oluşturur
    
    Args:
        image: BGR formatında buzdolabı görseli
        
    Returns:
        Binary mask (beyaz alanlar = 255, diğer alanlar = 0)
    """
    try:
        # BGR'dan HSV'ye dönüştür
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Beyaz renk aralığını tanımla
        lower_white = np.array([0, 0, 200])    # Alt sınır
        upper_white = np.array([180, 50, 255]) # Üst sınır
        
        # Beyaz alanlar için maske oluştur
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        
        return white_mask
        
    except Exception as e:
        print(f"Raf maskesi oluşturma hatası: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8)
