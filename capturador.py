# capturador.py
import time
import mss
from PIL import Image, ImageEnhance
import pytesseract

# Descomente e ajuste se o Tesseract não estiver no PATH do Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capturar_e_extrair_texto():
    """Captura a tela, redimensiona, melhora o contraste e extrai texto com OCR."""
    with mss.MSS() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Aumenta a resolução para o dobro (reduz ruído de fontes pequenas)
        largura, altura = img.size
        img = img.resize((largura * 2, altura * 2), Image.BICUBIC)
        
        # Aumenta nitidez para melhorar bordas das letras
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        
        # Converte para escala de cinza (OCR funciona melhor)
        img = img.convert('L')
        
        print("🖼️ Tela capturada e processada. Executando OCR...")
        # Usa --psm 6 (bloco uniforme de texto) que funciona melhor com UIs
        texto = pytesseract.image_to_string(img, lang='por', config='--psm 6')
        return texto

if __name__ == "__main__":
    texto_encontrado = capturar_e_extrair_texto()
    print("--- TEXTO ENCONTRADO NA TELA ---")
    print(texto_encontrado)
    print("-------------------------------")