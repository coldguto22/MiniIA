import json

import pytest

from benchmarks.compare_models import run_benchmark, write_report


class FakeClient:
    def list(self):
        return {"models": [{"name": name} for name in (
            "qwen2.5:3b", "llama3.2:latest"
        )]}

    def generate(self, model, prompt, **kwargs):
        return {
            "response": f"Resposta de {model}",
            "total_duration": 1_000_000,
            "load_duration": 100_000,
            "prompt_eval_duration": 200_000,
            "eval_duration": 500_000,
            "prompt_eval_count": 12,
            "eval_count": 8,
        }


JUDGMENT = json.dumps({
    "A": {"aderencia": 4, "correcao": 4, "clareza": 4, "completude": 4},
    "B": {"aderencia": 3, "correcao": 3, "clareza": 3, "completude": 3},
    "justificativa": "A resposta A atendeu melhor ao pedido.",
})


@pytest.mark.unit
def test_benchmark_reports_tokens_latency_and_blind_quality_scores():
    result = run_benchmark(
        client=FakeClient(),
        prompts=["Responda em português."],
        runs=1,
        warmups=0,
        judge=lambda _prompt: JUDGMENT,
    )

    assert result["summary"]["qwen"]["avg_input_tokens"] == 12
    assert result["summary"]["llama"]["avg_output_tokens"] == 8
    assert result["summary"]["qwen"]["avg_quality_score_1_to_5"] in (3, 4)
    assert result["cases"][0]["responses"]["qwen"]["response"] == "Resposta de qwen2.5:3b"


@pytest.mark.unit
def test_benchmark_writes_json_and_markdown_summary(tmp_path):
    result = run_benchmark(
        client=FakeClient(), prompts=["Uma pergunta"], warmups=0,
        judge=lambda _prompt: JUDGMENT,
    )
    json_path, markdown_path = write_report(result, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["method"]["judge_model"] == "llama3.2:latest"
    assert "Qualidade média" in markdown_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_benchmark_fails_clearly_when_a_model_is_missing():
    class MissingLlama(FakeClient):
        def list(self):
            return {"models": [{"name": "qwen2.5:3b"}]}

    with pytest.raises(RuntimeError, match="llama3.2:latest"):
        run_benchmark(client=MissingLlama(), prompts=["teste"], warmups=0)
