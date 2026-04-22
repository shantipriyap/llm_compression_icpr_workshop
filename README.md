# ICPR 2026 Compression Workshop

## Multilingual Robustness of Modern LLM Compression
**A Study on Qwen3, Gemma4, and Phi-4 Mini**

---

## Project Structure

```
icpr/
├── config/
│   ├── models.yaml           # Model registry (HF IDs, sizes, families)
│   ├── benchmarks.yaml       # Benchmark datasets & metrics config
│   └── compression.yaml      # Compression method parameters
│
├── compression/
│   ├── gptq_quantize.py      # GPTQ 4-bit quantization (Owner: Sakshi & Mahi)
│   ├── awq_quantize.py       # AWQ 4-bit quantization  (Owner: Subrat, Anurag)
│   └── kv_cache_compress.py  # KV-cache compression    (Owner: Debasish)
│
├── benchmarks/
│   ├── utils.py              # Shared model loading & generation helpers
│   ├── gsm8k_eval.py         # GSM8K evaluation (math QA)
│   ├── msmarco_eval.py       # MS MARCO evaluation (passage QA)
│   └── boolq_eval.py         # BoolQ evaluation (boolean QA)
│
├── evaluation/
│   ├── run_all.py            # Run all benchmarks for one model+compression
│   └── collect_results.py    # Aggregate results across all runs → CSV table
│
├── scripts/
│   ├── run_baseline.sh       # BF16 baseline runs
│   ├── run_gptq.sh           # GPTQ quantize + evaluate
│   ├── run_awq.sh            # AWQ quantize + evaluate
│   └── run_kv_compress.sh    # KV-cache compression + evaluate
│
├── results/                  # Output directory (auto-created)
├── quantized/                # Saved quantized checkpoints (auto-created)
├── requirements.txt
└── README.md
```

---

## Models

| Key | Model | Size | Family |
|---|---|---|---|
| `qwen3_8b` | Qwen/Qwen3-8B | 8B | Qwen3 |
| `qwen3_35b` | Qwen3.6-35B-A3B | 35B MoE | Qwen3 |
| `gemma4_12b` | Gemma4-12B | 12B | Gemma4 |
| `gemma4_27b` | Gemma4-27B | 27B | Gemma4 |
| `phi4_mini` | Phi-4-Mini | 3.8B | Phi-4 |

---

## Compression Methods

| Method | Script | Key Params |
|---|---|---|
| BF16 Baseline | — | full precision |
| GPTQ 4-bit | `compression/gptq_quantize.py` | bits=4, group=128 |
| AWQ 4-bit | `compression/awq_quantize.py` | bits=4, group=128 |
| KV-Cache Quant | `compression/kv_cache_compress.py` | kv_bits=4, residual=128 |

---

## Benchmarks

### English

| Benchmark | Task | Metric | Samples |
|---|---|---|---|
| GSM8K | Math QA | Exact Match | 1319 |
| MS MARCO | Passage QA | ROUGE-L | 1000 |
| BoolQ | Boolean QA | Accuracy | 3270 |

### Multilingual (Hindi + Odia)

| Benchmark | Language | Task | Metric | Dataset |
|---|---|---|---|---|
| MGSM | Hindi (hi) | Math reasoning | Exact Match | `juletxara/mgsm` |
| IndicQA | Hindi (hi) | Reading comprehension | F1 + EM | `ai4bharat/IndicQA` |
| IndicQA | Odia (or) | Reading comprehension | F1 + EM | `ai4bharat/IndicQA` |
| IndicSentiment | Hindi (hi) | Sentiment | Accuracy | `ai4bharat/IndicSentiment` |
| IndicSentiment | Odia (or) | Sentiment | Accuracy | `ai4bharat/IndicSentiment` |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run BF16 baseline (no quantization)
./scripts/run_baseline.sh

# 3. Run GPTQ experiments (Owner: Sakshi & Mahi)
./scripts/run_gptq.sh

# 4. Run AWQ experiments (Owner: Subrat, Anurag)
./scripts/run_awq.sh

# 5. Run KV-cache compression experiments (Owner: Debasish)
./scripts/run_kv_compress.sh

# 6. Collect and compare all results
python evaluation/collect_results.py --results-dir results/
```

**Quick test with fewer samples** (e.g., 50 per benchmark):
```bash
MAX_SAMPLES=50 ./scripts/run_baseline.sh
```

---

## Results

> Evaluated on 200-sample subsets. Full-set runs pending.  
> ✅ = complete | ⏳ = pending | — = not applicable

### Accuracy / ROUGE-L

| Model | Method | GSM8K ↑ | BoolQ ↑ | MS MARCO ROUGE-L ↑ | VRAM (GB) |
|---|---|---|---|---|---|
| Qwen3-8B | BF16 Baseline | **19.0%** | **87.5%** | **0.0616** | 15.3 |
| Qwen3-8B | KV-compress 4-bit | 19.0% | 87.5% | 0.0620 | 15.3 |
| Qwen3-8B | AWQ 4-bit | ⏳ | ⏳ | ⏳ | — |
| Qwen3-8B | GPTQ 4-bit | ⏳ | ⏳ | ⏳ | — |
| Phi-4-Mini | BF16 Baseline | ⏳ | ⏳ | ⏳ | — |
| Phi-4-Mini | KV-compress 4-bit | **85.5%** | **83.5%** | **0.1398** | 7.2 |
| Phi-4-Mini | AWQ 4-bit | ⏳ | ⏳ | ⏳ | — |
| Phi-4-Mini | GPTQ 4-bit | ⏳ | ⏳ | ⏳ | — |

### Degradation vs BF16 Baseline (Qwen3-8B)

| Method | GSM8K drop | BoolQ drop | ROUGE-L drop |
|---|---|---|---|
| KV-compress 4-bit | **0.0%** | **0.0%** | +0.65% (improved) |
| AWQ 4-bit | ⏳ | ⏳ | ⏳ |
| GPTQ 4-bit | ⏳ | ⏳ | ⏳ |

### Throughput (tokens/sec)

| Model | Method | GSM8K | BoolQ | MS MARCO |
|---|---|---|---|---|
| Qwen3-8B | BF16 Baseline | 59.4 | 54.9 | 58.9 |
| Qwen3-8B | KV-compress 4-bit | 59.4 | 53.7 | 57.5 |
| Phi-4-Mini | KV-compress 4-bit | 96.9 | 75.6 | 93.9 |

### Multilingual Results (Hindi + Odia) — Pending

| Model | Method | MGSM-Hi ↑ | IndicQA-Hi F1 ↑ | IndicQA-Or F1 ↑ | Sentiment-Hi ↑ | Sentiment-Or ↑ |
|---|---|---|---|---|---|---|
| Qwen3-8B | BF16 Baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Qwen3-8B | KV-compress 4-bit | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Qwen3-8B | AWQ 4-bit | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Qwen3-8B | GPTQ 4-bit | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Phi-4-Mini | BF16 Baseline | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Phi-4-Mini | KV-compress 4-bit | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Phi-4-Mini | AWQ 4-bit | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Phi-4-Mini | GPTQ 4-bit | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

> Quantized models pushed to HuggingFace: [shantipriya/Qwen3-8B-AWQ-4bit](https://huggingface.co/shantipriya/Qwen3-8B-AWQ-4bit)

---

## Output

Results are written to `results/<model_key>/<compression>/`:
- `gsm8k_results.json`
- `msmarco_results.json`
- `boolq_results.json`
- `summary.json`

Aggregate CSV: `results/all_results.csv`
