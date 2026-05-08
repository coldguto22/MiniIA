# capturador.py
import time
import mss
from PIL import Image
import pytesseract

# No Windows, descomente e ajuste a linha abaixo se o Tesseract não estiver no PATH.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capturar_e_extrair_texto():
    """Captura a tela toda e retorna o texto encontrado."""
    with mss.mss() as sct:
        # Captura o monitor principal (índice 1)
        screenshot = sct.grab(sct.monitors[1])
        # Converte para um objeto Image do Pillow
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # (Opcional) Redimensiona a imagem se for muito grande para acelerar
        # largura, altura = img.size
        # img = img.resize((largura // 2, altura // 2))

        print("🖼️ Tela capturada. Executando OCR...")
        # Extrai texto da imagem (português)
        texto = pytesseract.image_to_string(img, lang='por')
        return texto

if __name__ == "__main__":
    texto_encontrado = capturar_e_extrair_texto()
    print("--- TEXTO ENCONTRADO NA TELA ---")
    print(texto_encontrado)
    print("-------------------------------")