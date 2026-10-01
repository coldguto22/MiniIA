from __future__ import annotations

import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - optional only for PNG export
    Image = None
    ImageDraw = None

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency in constrained envs
    ollama = None

from dante.config import load_models_config

PROMPTS = [
    "Explique em duas frases por que a observação contínua ajuda uma IA a lembrar contexto.",
    "Descreva uma rotina de trabalho eficiente para um time de engenharia de IA.",
    "Escreva um resumo curto de um artigo sobre memória de longo prazo em modelos de linguagem.",
    "Dê um exemplo de prompt útil para avaliar coerência em respostas curtas.",
    "Explique o que é isolamento do modelo em um sistema multi-agente.",
    "Crie uma frase criativa sobre uma estação do ano e um editor de código.",
    "Como você descreveria a diferença entre atenção e estado recorrente em LLMs?",
    "Escreva um texto técnico sobre embeddings e recuperação por similaridade.",
    "Responda em português em três frases sobre um cenário de automação doméstica.",
    "Explique por que benchmarks devem incluir latência e qualidade em conjunto.",
    "Crie uma mensagem curta para um usuário que acabou de começar a usar um assistente local.",
    "Descreva em 2 linhas o trade-off entre velocidade e profundidade de raciocínio.",
    "Quais cuidados um sistema de observação contínua deve ter com privacidade?",
    "Escreva uma pergunta útil para avaliar a capacidade de reflexão de uma IA.",
    "Descreva uma experiência de uso de um diário humano em um sistema de IA.",
    "Explique em poucas palavras o que é um state space model.",
    "Crie uma resposta curta para uma tela de dashboard de produtividade.",
    "Escreva uma observação reflexiva sobre rotina, atenção e memória.",
    "Explique por que um fallback é importante em um pipeline de modelos locais.",
    "Resuma em duas frases como o contexto pode melhorar respostas de IA.",
]


def _ollama_ready() -> bool:
    if ollama is None:
        return False
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            return True
    except OSError:
        return False


def _installed_models() -> set[str]:
    if not _ollama_ready():
        return set()
    try:
        response = ollama.list()
    except Exception:
        return set()

    models = response.get("models", []) if isinstance(response, dict) else []
    names: set[str] = set()
    for item in models:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                names.add(name)
        elif hasattr(item, "name"):
            names.add(str(item.name))
    return names


def _safe_generate(model_name: str, prompt: str) -> tuple[str, float, int]:
    if model_name not in _installed_models():
        return (
            f"[simulado] modelo={model_name}; prompt_len={len(prompt)}",
            0.0,
            len(prompt),
        )

    start = time.perf_counter()
    try:
        client = getattr(ollama, "Client", None)
        if client is not None:
            response = client(timeout=5).generate(model=model_name, prompt=prompt)
        else:
            response = ollama.generate(model=model_name, prompt=prompt)
        text = str(response.get("response", "")).strip() if isinstance(response, dict) else str(response).strip()
        elapsed = time.perf_counter() - start
        return text, elapsed, len(text.split())
    except Exception as exc:  # pragma: no cover - network/model missing
        elapsed = time.perf_counter() - start
        return f"[falha no modelo {model_name}: {exc}]", elapsed, 0


def _generate_png(output_path: Path, payload: dict) -> None:
    if Image is None or ImageDraw is None:
        return
    image = Image.new("RGB", (840, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 830, 410), outline="black")
    draw.text((30, 20), "Benchmark System 1", fill="black")
    info = payload.get("summary", {})
    draw.text((30, 60), f"Qwen p95: {info.get('qwen_p95_ms', 0)} ms", fill="black")
    draw.text((30, 90), f"Mamba p95: {info.get('mamba_p95_ms', 0)} ms", fill="black")
    draw.text((30, 120), f"Qwen throughput: {info.get('qwen_throughput_tokens_per_s', 0)}", fill="black")
    draw.text((30, 150), f"Mamba throughput: {info.get('mamba_throughput_tokens_per_s', 0)}", fill="black")
    image.save(output_path)


def main() -> None:
    config = load_models_config()
    system1 = config["system1"]
    fallback = system1.get("fallback", "qwen2.5:3b")
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen_results = []
    mamba_results = []

    for prompt in PROMPTS:
        qwen_text, qwen_elapsed, qwen_tokens = _safe_generate(fallback, prompt)
        qwen_results.append({"prompt": prompt, "response": qwen_text, "elapsed_sec": qwen_elapsed, "tokens": qwen_tokens})

        mamba_text, mamba_elapsed, mamba_tokens = _safe_generate(system1["model"], prompt)
        mamba_results.append({"prompt": prompt, "response": mamba_text, "elapsed_sec": mamba_elapsed, "tokens": mamba_tokens})

    def summarize(results: list[dict]) -> dict:
        latencies = [entry["elapsed_sec"] * 1000 for entry in results]
        token_counts = [entry["tokens"] for entry in results]
        total_tokens = sum(token_counts)
        total_elapsed = sum(latencies) / 1000.0
        throughput = (total_tokens / total_elapsed) if total_elapsed else 0.0
        return {
            "p50_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p95_ms": sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0,
            "throughput_tokens_per_s": throughput,
            "avg_response_tokens": (total_tokens / len(results)) if results else 0,
        }

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": config,
        "systems": {
            "qwen": {"model": fallback, "results": qwen_results, "summary": summarize(qwen_results)},
            "mamba": {"model": system1["model"], "results": mamba_results, "summary": summarize(mamba_results)},
        },
    }
    payload["summary"] = {
        "qwen_p95_ms": payload["systems"]["qwen"]["summary"]["p95_ms"],
        "mamba_p95_ms": payload["systems"]["mamba"]["summary"]["p95_ms"],
        "qwen_throughput_tokens_per_s": payload["systems"]["qwen"]["summary"]["throughput_tokens_per_s"],
        "mamba_throughput_tokens_per_s": payload["systems"]["mamba"]["summary"]["throughput_tokens_per_s"],
    }

    timestamp = datetime.utcnow().strftime("%Y%m%d")
    output_path = output_dir / f"compare_{timestamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _generate_png(output_dir / "compare.png", payload)

    print(f"Benchmark salvo em {output_path}")
    print(f"Gráfico salvo em {output_dir / 'compare.png'}")


if __name__ == "__main__":
    main()
