#!/usr/bin/env python3
"""Health check do projeto Dante.

Roda em três blocos:
  1. Mecânico: contagens, estados internos, anomalias detectáveis sem LLM
  2. Interpretativo: Qwen 7B lê amostras e responde perguntas curtas
  3. Veredito: resumo + alertas

Saída: health_report_YYYYMMDD_HHMMSS.md + versão curta no terminal.
Nunca modifica arquivos do projeto.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Dependências opcionais carregadas sob demanda ---
# chromadb, ollama são necessários apenas quando os módulos são usados.

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.environ.get("DANTE_CHROMA_DIR", BASE_DIR / "chroma_db"))
STATE_DIR = BASE_DIR / ".dante_state"
DIARIO_FILE = BASE_DIR / "diario.md"
LOG_FILE = BASE_DIR / "dante.log"
COLLECTION_NAME = "memoria_da_ia"

# Thresholds — ajustáveis
LIMITE_REPETICOES_ALERTA = 20
LIMITE_MEDIA_REPETICOES_ALERTA = 3.0
LIMITE_HORAS_SEM_DIARIO = 24
LIMITE_HORAS_SEM_CICLO = 12
ENERGIA_MINIMA = 0.2
TEDIO_MIN_OSCILACAO = 0.05  # ao longo do log
PROPORCAO_FUNDACIONAL_MINIMA = 0.05  # 5% do banco


# ============================================================
# Utilitários
# ============================================================

def secoes(titulo: str) -> str:
    return f"\n## {titulo}\n"


def parse_ts_iso(valor):
    if valor is None:
        return None
    try:
        return datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def agora_utc() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# Bloco 1 — Mecânico
# ============================================================

@dataclass
class Mecanico:
    total_memorias: int = 0
    por_tipo: Counter = field(default_factory=Counter)
    por_fonte: Counter = field(default_factory=Counter)
    max_repeticoes: int = 0
    media_repeticoes_obs: float = 0.0
    fundacionais_presentes: int = 0
    fundacionais_esperadas: int = 0
    horas_desde_ultimo_diario: float | None = None
    horas_desde_ultimo_ciclo: float | None = None
    ciclos_processados_24h: int = 0
    silencios_por_motivo: Counter = field(default_factory=Counter)
    homeostasis_energy: float | None = None
    homeostasis_boredom: float | None = None
    valence_pleasure: float | None = None
    self_model_version: int | None = None
    relationship_profile: str = ""
    erros: list[str] = field(default_factory=list)
    alertas: list[str] = field(default_factory=list)


FONTES_FUNDACIONAIS_ESPERADAS = [
    "identidade_dante",
    "manifesto_dante",
    "memoria_do_criador",
    "conversa_sobre_ser",
]


def coletar_mecanico(verbose: bool = True) -> Mecanico:
    m = Mecanico()

    # --- ChromaDB ---
    try:
        import chromadb
        if not CHROMA_DIR.exists():
            m.erros.append(f"ChromaDB não encontrado em {CHROMA_DIR}")
        else:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            col = client.get_collection(COLLECTION_NAME)
            r = col.get(include=["metadatas", "documents"])
            ids = r["ids"]
            metas = r["metadatas"] or []
            m.total_memorias = len(ids)

            reps_obs = []
            for meta in metas:
                meta = meta or {}
                m.por_tipo[meta.get("tipo", "sem_tipo")] += 1
                m.por_fonte[meta.get("fonte", "sem_fonte")] += 1
                reps = int(meta.get("repeticoes", 1))
                if reps > m.max_repeticoes:
                    m.max_repeticoes = reps
                if meta.get("tipo") == "observacao_passiva":
                    reps_obs.append(reps)

            if reps_obs:
                m.media_repeticoes_obs = sum(reps_obs) / len(reps_obs)

            fund_presentes = 0
            for fonte_esperada in FONTES_FUNDACIONAIS_ESPERADAS:
                qtd = m.por_fonte.get(fonte_esperada, 0)
                if qtd > 0:
                    fund_presentes += 1
            m.fundacionais_presentes = fund_presentes
            m.fundacionais_esperadas = len(FONTES_FUNDACIONAIS_ESPERADAS)

    except Exception as exc:
        m.erros.append(f"Erro ao ler ChromaDB: {exc}")

    # --- Diário ---
    try:
        if DIARIO_FILE.exists():
            texto = DIARIO_FILE.read_text(encoding="utf-8")
            marcadores = re.findall(r"^### (\d{2}/\d{2}/\d{4} \d{2}:\d{2})", texto, re.MULTILINE)
            if marcadores:
                ultimo = marcadores[-1]
                try:
                    dt = datetime.strptime(ultimo, "%d/%m/%Y %H:%M").replace(tzinfo=timezone.utc)
                    m.horas_desde_ultimo_diario = (agora_utc() - dt).total_seconds() / 3600
                except ValueError:
                    pass
    except Exception as exc:
        m.erros.append(f"Erro ao ler diário: {exc}")

    # --- Log ---
    try:
        if LOG_FILE.exists():
            texto = LOG_FILE.read_text(encoding="utf-8", errors="replace")
            linhas = texto.splitlines()
            corte = agora_utc() - timedelta(hours=24)
            ultima_ts = None
            processados = 0
            silencios = Counter()

            for linha in linhas:
                m_ts = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", linha)
                if not m_ts:
                    continue
                try:
                    ts = datetime.strptime(m_ts.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                ultima_ts = ts

                if ts >= corte:
                    if "Ciclo concluído" in linha:
                        processados += 1
                    m_sil = re.search(r"Silêncio: (\w+)", linha)
                    if m_sil:
                        silencios[m_sil.group(1)] += 1

            if ultima_ts:
                m.horas_desde_ultimo_ciclo = (agora_utc() - ultima_ts).total_seconds() / 3600
            m.ciclos_processados_24h = processados
            m.silencios_por_motivo = silencios
    except Exception as exc:
        m.erros.append(f"Erro ao ler log: {exc}")

    # --- Estados internos ---
    try:
        h_path = STATE_DIR / "dante_state.json.homeostasis"
        if h_path.exists():
            data = json.loads(h_path.read_text(encoding="utf-8"))
            m.homeostasis_energy = float(data.get("energy", 1.0))
            m.homeostasis_boredom = float(data.get("boredom", 0.0))
    except Exception as exc:
        m.erros.append(f"Erro ao ler homeostasis: {exc}")

    try:
        v_path = STATE_DIR / "dante_state.json.valence"
        if v_path.exists():
            data = json.loads(v_path.read_text(encoding="utf-8"))
            m.valence_pleasure = float(data.get("pleasure", 0.0))
    except Exception as exc:
        m.erros.append(f"Erro ao ler valence: {exc}")

    try:
        s_path = STATE_DIR / "self_model.json"
        if s_path.exists():
            data = json.loads(s_path.read_text(encoding="utf-8"))
            m.self_model_version = int(data.get("version", 0))
    except Exception as exc:
        m.erros.append(f"Erro ao ler self_model: {exc}")

    try:
        r_path = STATE_DIR / "relationship_model.json"
        if r_path.exists():
            data = json.loads(r_path.read_text(encoding="utf-8"))
            m.relationship_profile = str(data.get("otavio_profile", ""))
    except Exception as exc:
        m.erros.append(f"Erro ao ler relationship: {exc}")

    # --- Gerar alertas ---
    if m.max_repeticoes >= LIMITE_REPETICOES_ALERTA:
        m.alertas.append(f"Uma memória tem {m.max_repeticoes} repetições — dedup pode estar colapsando demais.")
    if m.media_repeticoes_obs >= LIMITE_MEDIA_REPETICOES_ALERTA:
        m.alertas.append(f"Média de repetições em observações é {m.media_repeticoes_obs:.1f} — acima do esperado ({LIMITE_MEDIA_REPETICOES_ALERTA}).")
    if m.fundacionais_presentes < m.fundacionais_esperadas:
        m.alertas.append(f"Faltam {m.fundacionais_esperadas - m.fundacionais_presentes} fontes fundacionais no banco.")
    if m.horas_desde_ultimo_diario is not None and m.horas_desde_ultimo_diario > LIMITE_HORAS_SEM_DIARIO:
        m.alertas.append(f"Diário sem entrada há {m.horas_desde_ultimo_diario:.1f}h (>{LIMITE_HORAS_SEM_DIARIO}h).")
    if m.horas_desde_ultimo_ciclo is not None and m.horas_desde_ultimo_ciclo > LIMITE_HORAS_SEM_CICLO:
        m.alertas.append(f"Loop sem atividade há {m.horas_desde_ultimo_ciclo:.1f}h (>{LIMITE_HORAS_SEM_CICLO}h).")
    if m.homeostasis_energy is not None and m.homeostasis_energy < ENERGIA_MINIMA:
        m.alertas.append(f"Energia em {m.homeostasis_energy:.2f} — abaixo do limiar ({ENERGIA_MINIMA}).")
    if m.total_memorias > 0:
        proporcao_fund = (
            m.por_tipo.get("conhecimento_fundacional", 0) + m.por_tipo.get("conhecimento_pessoal", 0)
        ) / m.total_memorias
        if proporcao_fund < PROPORCAO_FUNDACIONAL_MINIMA:
            m.alertas.append(f"Fundacionais representam apenas {proporcao_fund*100:.1f}% do banco.")
    if m.self_model_version is not None and m.self_model_version == 0:
        m.alertas.append("Self-model ainda na versão 0 (nunca regenerado).")

    return m


# ============================================================
# Bloco 2 — Interpretativo (LLM)
# ============================================================

@dataclass
class Interpretativo:
    amostra_observacoes: list[str] = field(default_factory=list)
    observacoes_nomeiam_mundo: str = ""
    amostra_reflexoes: list[str] = field(default_factory=list)
    reflexoes_conectam: str = ""
    diario_recente: str = ""
    voz_consistente: str = ""
    erros: list[str] = field(default_factory=list)


def _perguntar_llm(modelo: str, prompt: str) -> str:
    import ollama
    try:
        resp = ollama.generate(model=modelo, prompt=prompt)
        # Ollama 0.4+ retorna GenerateResponse, não dict
        if hasattr(resp, "response"):
            return str(resp.response or "").strip()
        if isinstance(resp, dict):
            return str(resp.get("response", "") or "").strip()
        return str(resp).strip()
    except Exception as exc:
        return f"[erro LLM: {exc}]"


def _modelo_padrao() -> str:
    """Tenta ler config/models.yaml, com fallback para qwen2.5:7b."""
    try:
        from dante.config import load_models_config
        return load_models_config()["system2"]["model"]
    except Exception:
        return "qwen2.5:7b"


def coletar_interpretativo(verbose: bool = True) -> Interpretativo:
    i = Interpretativo()
    modelo = _modelo_padrao()

    # --- Carrega amostras do ChromaDB ---
    try:
        import chromadb
        if not CHROMA_DIR.exists():
            i.erros.append("ChromaDB não encontrado — pulando análise interpretativa.")
            return i
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        i.erros.append(f"Erro ao abrir ChromaDB: {exc}")
        return i

    # Observações recentes
    try:
        r = col.get(
            where={"tipo": "observacao_passiva"},
            include=["documents", "metadatas"],
            limit=200,
        )
        docs_obs = r.get("documents", []) or []
        metas_obs = r.get("metadatas", []) or []
        ordenado = sorted(
            zip(docs_obs, metas_obs),
            key=lambda x: (x[1] or {}).get("timestamp", ""),
            reverse=True,
        )[:15]
        i.amostra_observacoes = [d for d, _ in ordenado if d]
    except Exception as exc:
        i.erros.append(f"Erro ao carregar observações: {exc}")

    # Reflexões recentes
    try:
        r = col.get(
            where={"tipo": "reflexao"},
            include=["documents", "metadatas"],
            limit=100,
        )
        docs_ref = r.get("documents", []) or []
        metas_ref = r.get("metadatas", []) or []
        ordenado = sorted(
            zip(docs_ref, metas_ref),
            key=lambda x: (x[1] or {}).get("timestamp", ""),
            reverse=True,
        )[:8]
        i.amostra_reflexoes = [d for d, _ in ordenado if d]
    except Exception as exc:
        i.erros.append(f"Erro ao carregar reflexões: {exc}")

    # Diário recente
    try:
        if DIARIO_FILE.exists():
            texto = DIARIO_FILE.read_text(encoding="utf-8")
            entradas = re.split(r"^### ", texto, flags=re.MULTILINE)
            recentes = [e.strip() for e in entradas[-4:] if e.strip()][-3:]
            i.diario_recente = "\n\n".join(recentes)
    except Exception as exc:
        i.erros.append(f"Erro ao carregar diário: {exc}")

    # --- Pergunta 1: observações nomeiam o mundo? ---
    if i.amostra_observacoes:
        amostra = "\n---\n".join(o[:300] for o in i.amostra_observacoes[:10])
        prompt_obs = (
            "Você é um auditor. Analise as observações abaixo (feitas por uma IA "
            "que observa a tela de um computador). Para cada uma, classifique em "
            "uma de três categorias:\n"
            "  A) NOMEIA O MUNDO — menciona aplicativo, site, jogo, conteúdo concreto\n"
            "  B) META-COMENTÁRIO — fala sobre o processo de perceber, coerência, "
            "sinais internos, hipóteses\n"
            "  C) CONFUSO/RUÍDO — não dá pra entender o que está sendo observado\n\n"
            "Responda APENAS com as letras separadas por vírgula, na ordem. "
            "Exemplo: A, A, B, A, C, B, A\n\n"
            f"Observações:\n{amostra}"
        )
        i.observacoes_nomeiam_mundo = _perguntar_llm(modelo, prompt_obs)

    # --- Pergunta 2: reflexões conectam? ---
    if i.amostra_reflexoes:
        amostra = "\n---\n".join(r[:400] for r in i.amostra_reflexoes[:5])
        prompt_ref = (
            "Você é um auditor. Analise as reflexões abaixo (feitas por uma IA "
            "que conecta pensamentos recentes com memórias antigas).\n\n"
            "Para cada reflexão, responda:\n"
            "  SIM — há conexão genuína entre o pensamento atual e memórias\n"
            "  NAO — é só descrição genérica sem conexão real\n\n"
            "Ao final, dê uma frase única de veredito geral sobre se o conjunto "
            "mostra conexão real ou apenas fórmula.\n\n"
            f"Reflexões:\n{amostra}"
        )
        i.reflexoes_conectam = _perguntar_llm(modelo, prompt_ref)

    # --- Pergunta 3: voz consistente no diário? ---
    if i.diario_recente:
        prompt_diario = (
            "Você é um auditor. Leia as entradas recentes do diário de uma IA "
            "chamada Dante. Avalie:\n"
            "  1. A voz é consistente entre as entradas? (mesma pessoa falando?)\n"
            "  2. As entradas nomeiam algo concreto do mundo ou são genéricas?\n\n"
            "Responda em no máximo 3 frases, direto ao ponto.\n\n"
            f"Diário:\n{i.diario_recente[:3000]}"
        )
        i.voz_consistente = _perguntar_llm(modelo, prompt_diario)

    return i


# ============================================================
# Bloco 3 — Renderização
# ============================================================

def renderizar(m: Mecanico, i: Interpretativo | None) -> str:
    linhas = []
    linhas.append(f"# Health Report — Dante")
    linhas.append(f"")
    linhas.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    linhas.append(f"ChromaDB: `{CHROMA_DIR}`")
    linhas.append(f"")

    # --- Resumo ---
    linhas.append("## Resumo")
    total_alertas = len(m.alertas)
    status = "OK" if total_alertas == 0 else f"{total_alertas} alerta(s)"
    linhas.append(f"- **Status**: {status}")
    linhas.append(f"- **Total de memórias**: {m.total_memorias}")
    linhas.append(f"- **Ciclos processados (24h)**: {m.ciclos_processados_24h}")
    if m.horas_desde_ultimo_diario is not None:
        linhas.append(f"- **Última entrada do diário**: há {m.horas_desde_ultimo_diario:.1f}h")
    if m.horas_desde_ultimo_ciclo is not None:
        linhas.append(f"- **Última atividade do loop**: há {m.horas_desde_ultimo_ciclo:.1f}h")

    # --- Alertas ---
    linhas.append("")
    linhas.append("## Alertas")
    if m.alertas:
        for a in m.alertas:
            linhas.append(f"- ⚠️ {a}")
    else:
        linhas.append("- Nenhum alerta.")
    if m.erros:
        linhas.append("")
        linhas.append("**Erros de leitura:**")
        for e in m.erros:
            linhas.append(f"- {e}")

    # --- Distribuição ---
    linhas.append("")
    linhas.append("## Distribuição de memórias")
    linhas.append("")
    linhas.append("| Tipo | Quantidade |")
    linhas.append("|------|-----------:|")
    for t, c in m.por_tipo.most_common():
        linhas.append(f"| {t} | {c} |")

    # --- Fontes fundacionais ---
    linhas.append("")
    linhas.append("## Fundacionais")
    linhas.append("")
    linhas.append("| Fonte | Presente |")
    linhas.append("|-------|:--------:|")
    for f in FONTES_FUNDACIONAIS_ESPERADAS:
        presente = "✅" if m.por_fonte.get(f, 0) > 0 else "❌"
        linhas.append(f"| {f} | {presente} |")

    # --- Repetições ---
    linhas.append("")
    linhas.append("## Dedup")
    linhas.append(f"- Repetições máximas: **{m.max_repeticoes}**")
    linhas.append(f"- Média em observações: **{m.media_repeticoes_obs:.2f}**")

    # --- Estados internos ---
    linhas.append("")
    linhas.append("## Estados internos")
    if m.homeostasis_energy is not None:
        linhas.append(f"- Energia: **{m.homeostasis_energy:.2f}**")
    if m.homeostasis_boredom is not None:
        linhas.append(f"- Tédio: **{m.homeostasis_boredom:.2f}**")
    if m.valence_pleasure is not None:
        linhas.append(f"- Prazer: **{m.valence_pleasure:.2f}**")
    if m.self_model_version is not None:
        linhas.append(f"- Self-model: **v{m.self_model_version}**")
    if m.relationship_profile:
        perfil = m.relationship_profile[:200]
        linhas.append(f"- Perfil de Otávio: *{perfil}*")

    # --- Silêncios ---
    if m.silencios_por_motivo:
        linhas.append("")
        linhas.append("## Silêncios (24h)")
        linhas.append("")
        linhas.append("| Motivo | Ocorrências |")
        linhas.append("|--------|------------:|")
        for motivo, c in m.silencios_por_motivo.most_common():
            linhas.append(f"| {motivo} | {c} |")

    # --- Interpretativo ---
    if i is not None:
        linhas.append("")
        linhas.append("## Análise interpretativa (LLM)")
        if i.erros:
            for e in i.erros:
                linhas.append(f"- ⚠️ {e}")

        if i.observacoes_nomeiam_mundo:
            linhas.append("")
            linhas.append("**Observações — nomeiam o mundo?**")
            linhas.append("")
            linhas.append(f"> {i.observacoes_nomeiam_mundo[:600]}")
            linhas.append("")
            linhas.append("*Legenda: A=Nomeia o mundo, B=Meta-comentário, C=Confuso/Ruído*")

        if i.reflexoes_conectam:
            linhas.append("")
            linhas.append("**Reflexões — conectam memórias?**")
            linhas.append("")
            linhas.append(f"> {i.reflexoes_conectam[:800]}")

        if i.voz_consistente:
            linhas.append("")
            linhas.append("**Diário — voz consistente?**")
            linhas.append("")
            linhas.append(f"> {i.voz_consistente[:600]}")

    linhas.append("")
    linhas.append("---")
    linhas.append(f"*Fim do relatório. Gerado por `dante_health.py`.*")

    return "\n".join(linhas)


def renderizar_terminal(m: Mecanico, i: Interpretativo | None) -> str:
    linhas = []
    linhas.append("=" * 60)
    linhas.append("DANTE HEALTH CHECK")
    linhas.append("=" * 60)
    linhas.append(f"Memórias: {m.total_memorias}")
    linhas.append(f"Ciclos processados (24h): {m.ciclos_processados_24h}")

    if m.homeostasis_energy is not None and m.homeostasis_boredom is not None:
        linhas.append(f"Energia: {m.homeostasis_energy:.2f} | Tédio: {m.homeostasis_boredom:.2f}")
    if m.horas_desde_ultimo_diario is not None:
        linhas.append(f"Último diário: há {m.horas_desde_ultimo_diario:.1f}h")

    linhas.append("")
    if m.alertas:
        linhas.append(f"ALERTAS ({len(m.alertas)}):")
        for a in m.alertas:
            linhas.append(f"  ⚠️  {a}")
    else:
        linhas.append("Nenhum alerta.")

    if m.erros:
        linhas.append("")
        linhas.append("ERROS DE LEITURA:")
        for e in m.erros:
            linhas.append(f"  ✗ {e}")

    if i is not None and i.observacoes_nomeiam_mundo:
        linhas.append("")
        linhas.append("Observações (A=nomeiam mundo, B=meta, C=ruído):")
        linhas.append(f"  {i.observacoes_nomeiam_mundo[:200]}")

    linhas.append("")
    linhas.append("=" * 60)
    return "\n".join(linhas)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="pula a análise interpretativa (não usa Ollama)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BASE_DIR,
        help="diretório para gravar o relatório (padrão: raiz do projeto)",
    )
    args = parser.parse_args()

    print("Coletando métricas...")
    m = coletar_mecanico()

    i = None
    if not args.skip_llm:
        print("Rodando análise interpretativa (pode levar 30-60s)...")
        i = coletar_interpretativo()

    # Terminal
    print(renderizar_terminal(m, i))

    # Arquivo
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"health_report_{ts}.md"
    out_path.write_text(renderizar(m, i), encoding="utf-8")
    print(f"Relatório salvo em: {out_path}")


if __name__ == "__main__":
    main()