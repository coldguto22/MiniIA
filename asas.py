# asas.py
"""
Módulo de busca autônoma para o Dante.
Usa DuckDuckGo (via ddgs) para pesquisar e retornar snippets.
"""

from ddgs import DDGS

def pesquisar(query, max_results=3):
    """
    Realiza uma busca no DuckDuckGo e retorna uma lista de dicionários
    com 'title', 'url' e 'snippet'.
    """
    try:
        with DDGS() as ddgs:
            resultados = []
            # A API do ddgs retorna geradores; iteramos para obter os resultados
            for r in ddgs.text(query, max_results=max_results):
                resultados.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
            return resultados
    except Exception as e:
        print(f"[Asas] Erro na busca: {e}")
        return []

def formatar_resultados(resultados):
    """Formata os resultados da busca em uma string legível."""
    if not resultados:
        return "Nenhum resultado encontrado."
    
    texto = ""
    for i, r in enumerate(resultados, 1):
        texto += f"{i}. {r['title']}\n"
        texto += f"   {r['snippet'][:200]}\n"
        texto += f"   URL: {r['url']}\n\n"
    return texto.strip()