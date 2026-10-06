# capturador.py
def capturar_e_extrair_texto():
    """Compatibilidade com os módulos antigos; implementação em dante.perception."""
    from dante.perception.ocr import capture_and_extract_text

    print("🖼️ Tela capturada e processada. Executando OCR...")
    return capture_and_extract_text()

if __name__ == "__main__":
    texto_encontrado = capturar_e_extrair_texto()
    print("--- TEXTO ENCONTRADO NA TELA ---")
    print(texto_encontrado)
    print("-------------------------------")
