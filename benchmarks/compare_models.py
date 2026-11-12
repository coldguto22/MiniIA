"""Compara Qwen 2.5 3B e Llama 3.2 no Ollama local.

O benchmark mede tokens reportados pelo Ollama e latência de geração. A nota
de qualidade é uma avaliação assistida pelo Llama 3.2, útil para triagem, mas
não substitui uma avaliação humana cega.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import ollama

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_system1 import PROMPTS

DEFAULT_OUTPUT = ROOT / "benchmarks" / "results"
MODELS = {"qwen": "qwen2.5:3b", "llama": "llama3.2:latest"}
JUDGE_MODEL = "llama3.2:latest"
QUALITY_DIMENSIONS = ("aderencia", "correcao", "clareza", "completude")
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{
            label: {
                "type": "object",
                "properties": {
                    key: {"type": "integer", "minimum": 1, "maximum": 5}
                    for key in QUALITY_DIMENSIONS
                },
                "required": list(QUALITY_DIMENSIONS),
                "additionalProperties": False,
            }
            for label in ("A", "B")
        },
        "justificativa": {"type": "string"},
    },
    "required": ["A", "B", "justificativa"],
    "additionalProperties": False,
}


def _value(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def _model_names(client: Any) -> set[str]:
    result = client.list()
    items = result.get("models", []) if isinstance(result, dict) else result.models
    names = set()
    for item in items:
        name = (
            item.get("name") or item.get("model")
            if isinstance(item, dict)
            else getattr(item, "name", None) or getattr(item, "model", None)
        )
        if name:
            names.add(str(name))
    return names


def _generate(client: Any, model: str, prompt: str, *, num_predict: int) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.1, "seed": 42, "num_predict": num_predict},
    )
    wall_ms = (time.perf_counter() - start) * 1000
    return {
        "response": str(_value(response, "response", "")).strip(),
        "wall_latency_ms": round(wall_ms, 2),
        "ollama_total_duration_ms": round((_value(response, "total_duration", 0) or 0) / 1e6, 2),
        "load_duration_ms": round((_value(response, "load_duration", 0) or 0) / 1e6, 2),
        "prompt_eval_duration_ms": round((_value(response, "prompt_eval_duration", 0) or 0) / 1e6, 2),
        "eval_duration_ms": round((_value(response, "eval_duration", 0) or 0) / 1e6, 2),
        "input_tokens": _value(response, "prompt_eval_count"),
        "output_tokens": _value(response, "eval_count"),
    }


def _judge_prompt(prompt: str, answer_a: str, answer_b: str) -> str:
    return f"""Avalie duas respostas à mesma solicitação. Ignore qual modelo pode tê-las produzido.
Dê notas inteiras de 1 a 5 para aderencia (cumpre formato e instruções), correcao
(factualidade e ausência de invenções), clareza e completude (responde o suficiente
sem excesso). Use 1=ruim, 3=adequada, 5=excelente. Retorne somente JSON com as chaves
"A" e "B", cada uma contendo aderencia, correcao, clareza e completude como
notas numéricas de 1 a 5, e "justificativa" com uma frase curta comparativa.
Quando a solicitação fornecer uma fonte (por exemplo, texto obtido por OCR), use-a
como evidência principal e penalize afirmações não sustentadas por ela. Não premie
comprimento por si só.

Solicitação: {prompt}

Resposta A:
{answer_a[:3000]}

Resposta B:
{answer_b[:3000]}
"""


def _parse_judgment(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Juiz não retornou JSON válido: {raw[:200]}")
    payload = json.loads(raw[start : end + 1])
    scores = {}
    for label in ("A", "B"):
        values = payload[label]
        scores[label] = {
            key: max(1, min(5, int(values[key]))) for key in QUALITY_DIMENSIONS
        }
        scores[label]["media"] = round(statistics.mean(scores[label].values()), 2)
    return {"scores": scores, "justificativa": str(payload.get("justificativa", ""))}


def _summary(samples: list[dict[str, Any]], judge_scores: list[float]) -> dict[str, Any]:
    latencies = [item["wall_latency_ms"] for item in samples]
    input_tokens = [item["input_tokens"] for item in samples if item["input_tokens"] is not None]
    output_tokens = [item["output_tokens"] for item in samples if item["output_tokens"] is not None]
    return {
        "runs": len(samples),
        "avg_wall_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p50_wall_latency_ms": round(statistics.median(latencies), 2) if latencies else None,
        "p95_wall_latency_ms": round(sorted(latencies)[min(len(latencies)-1, int(len(latencies)*0.95))], 2) if latencies else None,
        "avg_input_tokens": round(statistics.mean(input_tokens), 2) if input_tokens else None,
        "avg_output_tokens": round(statistics.mean(output_tokens), 2) if output_tokens else None,
        "total_input_tokens": sum(input_tokens) if input_tokens else None,
        "total_output_tokens": sum(output_tokens) if output_tokens else None,
        "avg_quality_score_1_to_5": round(statistics.mean(judge_scores), 2) if judge_scores else None,
    }


def run_benchmark(
    *,
    client: Any | None = None,
    prompts: list[str] | None = None,
    runs: int = 1,
    warmups: int = 1,
    num_predict: int = 256,
    judge: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    client = client or ollama
    prompts = prompts or PROMPTS
    installed = _model_names(client)
    missing = [model for model in (*MODELS.values(), JUDGE_MODEL) if model not in installed]
    if missing:
        raise RuntimeError(f"Modelos ausentes no Ollama: {', '.join(missing)}")

    for model in MODELS.values():
        for _ in range(warmups):
            _generate(client, model, "Responda apenas: pronto.", num_predict=16)

    judge_fn = judge or (lambda prompt: str(_value(client.generate(
        model=JUDGE_MODEL,
        prompt=prompt,
        format=JUDGE_SCHEMA,
        options={"temperature": 0, "seed": 42, "num_predict": 180},
    ), "response", "")))
    rng = random.Random(42)
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in MODELS}
    cases = []

    for prompt_index, prompt in enumerate(prompts):
        for run_index in range(runs):
            # Alterna a ordem para reduzir o viés térmico por posição.
            order = list(MODELS)
            if (prompt_index + run_index) % 2:
                order.reverse()
            current = {}
            for key in order:
                current[key] = _generate(
                    client, MODELS[key], prompt, num_predict=num_predict
                )
                samples[key].append(current[key])

            labels = ["A", "B"]
            rng.shuffle(labels)
            key_to_label = dict(zip(MODELS, labels))
            label_to_key = {label: key for key, label in key_to_label.items()}
            judge_input = _judge_prompt(
                prompt,
                current[label_to_key["A"]]["response"],
                current[label_to_key["B"]]["response"],
            )
            try:
                judge_response = judge_fn(judge_input)
                judgment = _parse_judgment(judge_response)
                mapped_scores = {
                    key: judgment["scores"][key_to_label[key]] for key in MODELS
                }
                judge_error = None
            except Exception as exc:
                mapped_scores = {key: None for key in MODELS}
                judgment = {"justificativa": ""}
                judge_error = str(exc)

            cases.append({
                "prompt": prompt,
                "run": run_index + 1,
                "model_order": order,
                "responses": current,
                "quality": mapped_scores,
                "quality_rationale": judgment.get("justificativa", ""),
                "judge_response": judge_response if judge_error is None else None,
                "judge_error": judge_error,
            })

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {
            "models": MODELS,
            "judge_model": JUDGE_MODEL,
            "quality_dimensions": list(QUALITY_DIMENSIONS),
            "quality_note": "Notas de Llama 3.2 como juiz; triagem subjetiva, revisar respostas brutas antes de concluir.",
            "warmups_per_model": warmups,
            "runs_per_prompt": runs,
            "num_predict": num_predict,
            "latency": "parede (cliente) e durações reportadas pelo Ollama; inclui load_duration nos campos separados",
            "token_source": "prompt_eval_count/eval_count reportados pelo Ollama; não estimados por palavras",
        },
        "summary": {},
        "cases": cases,
    }
    for key in MODELS:
        scores = [case["quality"][key]["media"] for case in cases if case["quality"][key]]
        result["summary"][key] = {
            "model": MODELS[key],
            **_summary(samples[key], scores),
        }
    return result


def write_report(
    result: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    prefix: str = "qwen_llama",
    title: str = "Comparação Qwen 2.5:3b × Llama 3.2:latest",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{prefix}_{stamp}.json"
    md_path = output_dir / f"{prefix}_{stamp}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        f"Gerado em: {result['generated_at']}",
        "",
        "| Métrica | Qwen 2.5:3b | Llama 3.2:latest |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Latência média (ms)", "avg_wall_latency_ms"),
        ("Latência p95 (ms)", "p95_wall_latency_ms"),
        ("Tokens de entrada médios", "avg_input_tokens"),
        ("Tokens de saída médios", "avg_output_tokens"),
        ("Tokens de saída totais", "total_output_tokens"),
        ("Qualidade média (1–5)", "avg_quality_score_1_to_5"),
    ):
        q, llama = result["summary"]["qwen"].get(key), result["summary"]["llama"].get(key)
        lines.append(f"| {label} | {q if q is not None else 'n/d'} | {llama if llama is not None else 'n/d'} |")
    lines += [
        "",
        "> Qualidade foi pontuada pelo Llama 3.2 com respostas anonimizadas e ordem alternada. É uma avaliação auxiliar; consulte todas as respostas e justificativas no JSON.",
        "",
        "O JSON contém texto integral de cada resposta, métricas por chamada, notas por critério e justificativa do juiz.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="Repetições por prompt/modelo")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs deve ser >= 1 e --warmups deve ser >= 0")
    result = run_benchmark(runs=args.runs, warmups=args.warmups, num_predict=args.num_predict)
    json_path, md_path = write_report(result, args.output_dir)
    print(f"Resultados: {json_path}\nResumo: {md_path}")


if __name__ == "__main__":
    main()
