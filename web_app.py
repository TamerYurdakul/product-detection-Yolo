from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import os
import time
import logging
from analiz import raf_analizi_yap
import numpy as np

# FastAPI uygulaması
app = FastAPI(title="Ürün Tanıma Sistemi")

# Static dosyalar ve templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dizinler
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

# Static dizini oluştur
os.makedirs(STATIC_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request, 
    file: UploadFile = File(...),
    enhance: bool = Form(False),
    use_ensemble: bool = Form(False),

):
    try:
        # Dosya okuma
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # RGB'ye çevir
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # NumPy array'e çevir
        np_rgb = np.array(pil_image)
        
        logger.info(f"Analiz başlıyor... Kontrast: {enhance}, Ensemble: {use_ensemble}")
        
        # Basit analiz
        sonuc = raf_analizi_yap(np_rgb, enhance=enhance, use_ensemble=use_ensemble)
        logger.info("Analiz tamamlandı.")
        
        if isinstance(sonuc, dict) and "error" in sonuc:
            logger.error(f"Analiz hata: {sonuc['error']}")
            return templates.TemplateResponse(
                "index.html", 
                {"request": request, "error": sonuc["error"]}
            )
        
        # Analizden gelen raf bazlı sonuçları doğrudan kullan
        raf_listesi = sonuc.get("raf_bilgileri", [])
        toplam_urun = sonuc.get("toplam_urun", 0)
        gorsel_rgb = sonuc.get("gorsel")

        # Şablon doğrudan raf_listesi üzerinde dönecek
        raf_render_list = raf_listesi
        
        # Görseli kaydet
        out_name = f"analiz_{int(time.time())}.jpg"
        out_path = os.path.join(STATIC_DIR, out_name)
        
        try:
            Image.fromarray(gorsel_rgb).save(out_path, format="JPEG")
            logger.info(f"İşlenmiş görsel kaydedildi: {out_path}")
        except Exception as e:
            logger.exception("İşlenmiş görsel kaydedilemedi")
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Analiz görseli kaydedilemedi."}
            )
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": raf_render_list,
                "toplam_urun": toplam_urun,
                "image_url": f"/static/{out_name}",
            }
        )
        
    except Exception as e:
        logger.exception("Beklenmeyen hata")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Beklenmeyen hata: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="127.0.0.1", port=8001, reload=True)


