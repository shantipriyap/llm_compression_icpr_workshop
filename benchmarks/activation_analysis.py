"""
activation_analysis.py — Per-language activation statistics for LLMs.

Motivation (from SpQR / Wanda / ReLU-Strikes-Back literature):
  A small fraction of LLM weights act as high-impact outliers. Their
  activation patterns differ across input languages. This script measures
  per-layer L1 activation norms and sparsity for English vs. Indic-script
  inputs, providing empirical evidence for the hypothesis that compression
  disproportionately affects low-resource languages.

Usage:
  python benchmarks/activation_analysis.py \
      --model Qwen/Qwen3-8B \
      --output results/activation_stats.json

  Optional flags:
      --quantize_kv          # analyse KV-Quant 4-bit variant
      --num_samples 32       # prompts per language (default 16)
      --plot                 # save matplotlib figure alongside JSON
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Static sample prompts ─────────────────────────────────────────────────────

PROMPTS: Dict[str, List[str]] = {
    "en": [
        "What is the capital of France?",
        "Solve: 2 + 2 = ?",
        "Translate 'hello' to Spanish.",
        "Who wrote Hamlet?",
        "What is photosynthesis?",
        "Name three planets in the solar system.",
        "What is machine learning?",
        "When did World War II end?",
        "What is the speed of light?",
        "Describe the water cycle.",
        "What is democracy?",
        "Who invented the telephone?",
        "What is a neural network?",
        "Give the chemical formula for water.",
        "What is the square root of 144?",
        "Name the largest ocean.",
    ],
    "hi": [
        "फ्रांस की राजधानी क्या है?",
        "2 + 2 = ? हल करें।",
        "प्रकाश संश्लेषण क्या है?",
        "भारत की जनसंख्या क्या है?",
        "हिंदी में 'धन्यवाद' का अर्थ क्या है?",
        "मशीन लर्निंग क्या है?",
        "पानी का रासायनिक सूत्र क्या है?",
        "महात्मा गांधी कौन थे?",
        "सूर्य पृथ्वी से कितना दूर है?",
        "लोकतंत्र क्या होता है?",
        "तीन ग्रहों के नाम बताइए।",
        "हैमलेट किसने लिखा?",
        "न्यूरल नेटवर्क क्या है?",
        "पानी का चक्र क्या है?",
        "144 का वर्गमूल क्या है?",
        "सबसे बड़ा महासागर कौन सा है?",
    ],
    "or": [
        "ଫ୍ରାନ୍ସର ରାଜଧାନୀ କ'ଣ?",
        "2 + 2 = ? ସମାଧାନ କରନ୍ତୁ।",
        "ଆଲୋକ ସଂଶ୍ଲେଷଣ କ'ଣ?",
        "ଭାରତର ଜନସଂଖ୍ୟା କ'ଣ?",
        "ଓଡ଼ିଆ ଭାଷାରେ 'ଧନ୍ୟବାଦ' ର ଅର୍ଥ କ'ଣ?",
        "ମେସିନ୍ ଲର୍ନିଂ କ'ଣ?",
        "ଜଳର ରାସାୟନିକ ସୂତ୍ର କ'ଣ?",
        "ମହାତ୍ମା ଗାନ୍ଧୀ କିଏ ଥିଲେ?",
        "ସୂର୍ଯ୍ୟ ପୃଥିବୀ ଠାରୁ କେତେ ଦୂରରେ?",
        "ଗଣତନ୍ତ୍ର କ'ଣ?",
        "ତିନୋଟି ଗ୍ରହର ନାମ ଦିଅ।",
        "ହ୍ୟାମଲେଟ୍ କିଏ ଲେଖିଥିଲେ?",
        "ନ୍ୟୁରାଲ ନେଟୱାର୍କ କ'ଣ?",
        "ଜଳ ଚକ୍ର କ'ଣ?",
        "144 ର ବର୍ଗମୂଳ କ'ଣ?",
        "ସବୁଠୁ ବଡ଼ ମହାସାଗର କ'ଣ?",
    ],
    "bn": [
        "ফ্রান্সের রাজধানী কী?",
        "2 + 2 = ? সমাধান করুন।",
        "সালোকসংশ্লেষণ কী?",
        "ভারতের জনসংখ্যা কত?",
        "বাংলায় 'ধন্যবাদ' এর অর্থ কী?",
        "মেশিন লার্নিং কী?",
        "পানির রাসায়নিক সূত্র কী?",
        "মহাত্মা গান্ধী কে ছিলেন?",
        "সূর্য পৃথিবী থেকে কত দূরে?",
        "গণতন্ত্র কী?",
        "তিনটি গ্রহের নাম বল।",
        "হ্যামলেট কে লিখেছিলেন?",
        "নিউরাল নেটওয়ার্ক কী?",
        "জলচক্র কী?",
        "144-এর বর্গমূল কী?",
        "সবচেয়ে বড় মহাসাগর কোনটি?",
    ],
}

# ── Hook helpers ───────────────────────────────────────────────────────────────

class ActivationCollector:
    """Collects per-layer post-MLP activation statistics via forward hooks."""

    def __init__(self):
        self.handles: list = []
        self.stats: Dict[str, Dict] = {}  # layer_name -> {l1_norms, sparsity}

    def register(self, model: torch.nn.Module):
        """Attach a hook to every Linear layer whose name contains 'mlp'."""
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "mlp" in name:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            with torch.no_grad():
                act = output.float()  # (batch, seq, hidden) or (batch, hidden)
                flat = act.reshape(-1)
                l1 = flat.abs().mean().item()
                sparsity = (flat.abs() < 1e-6).float().mean().item()
                if name not in self.stats:
                    self.stats[name] = {"l1_norms": [], "sparsity": []}
                self.stats[name]["l1_norms"].append(l1)
                self.stats[name]["sparsity"].append(sparsity)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def aggregate(self) -> Dict[str, Dict]:
        """Return mean L1 norm and sparsity per layer."""
        return {
            name: {
                "mean_l1": float(np.mean(vals["l1_norms"])),
                "mean_sparsity": float(np.mean(vals["sparsity"])),
            }
            for name, vals in self.stats.items()
        }


# ── Main ───────────────────────────────────────────────────────────────────────

def load_model(model_name: str, quantize_kv: bool, hf_token: str | None):
    print(f"Loading model: {model_name}  quantize_kv={quantize_kv}")
    common_kwargs = dict(
        trust_remote_code=True,
        token=hf_token,
        device_map="auto",
    )
    if quantize_kv:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        common_kwargs["quantization_config"] = quant_config
    else:
        common_kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
    model.eval()
    return tokenizer, model


def run_prompts(
    prompts: List[str],
    tokenizer,
    model: torch.nn.Module,
    collector: ActivationCollector,
    max_new_tokens: int = 1,
):
    """Forward-pass each prompt; hooks collect activation statistics."""
    device = next(model.parameters()).device
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = enc.input_ids.to(device)
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)


def plot_results(agg: Dict[str, Dict[str, Dict]], output_path: str):
    """
    agg: { lang -> { layer_name -> {mean_l1, mean_sparsity} } }
    Plots mean L1 norm and sparsity across layers per language.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"en": "steelblue", "hi": "darkorange", "or": "green", "bn": "red"}

    for lang, layer_stats in agg.items():
        layers = list(layer_stats.keys())
        l1_vals = [layer_stats[l]["mean_l1"] for l in layers]
        sp_vals = [layer_stats[l]["mean_sparsity"] for l in layers]
        xs = range(len(layers))
        axes[0].plot(xs, l1_vals, label=lang, color=colors.get(lang, "gray"), alpha=0.8)
        axes[1].plot(xs, sp_vals, label=lang, color=colors.get(lang, "gray"), alpha=0.8)

    axes[0].set_title("Mean L1 Activation Norm per MLP Layer")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Mean |activation|")
    axes[0].legend()

    axes[1].set_title("Activation Sparsity per MLP Layer (|x| < 1e-6)")
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("Fraction near-zero")
    axes[1].legend()

    fig.suptitle("Per-language Activation Analysis (MLP layers)", fontsize=13)
    fig.tight_layout()
    plot_path = output_path.replace(".json", "_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-language activation analysis")
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--output", default="results/activation_stats.json")
    parser.add_argument("--quantize_kv", action="store_true",
                        help="Load model in 4-bit NF4 (tests compressed variant)")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of prompts per language (max 16)")
    parser.add_argument("--languages", nargs="+", default=["en", "hi", "or", "bn"],
                        choices=list(PROMPTS.keys()))
    parser.add_argument("--plot", action="store_true", help="Save matplotlib figure")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer, model = load_model(args.model, args.quantize_kv, hf_token)

    aggregated: Dict[str, Dict] = {}

    for lang in args.languages:
        prompts = PROMPTS[lang][: args.num_samples]
        collector = ActivationCollector()
        collector.register(model)

        print(f"Running {len(prompts)} prompts for language: {lang}")
        run_prompts(prompts, tokenizer, model, collector)
        collector.remove()

        aggregated[lang] = collector.aggregate()
        print(f"  {lang}: {len(aggregated[lang])} layers captured")

    # Compute summary statistics
    summary = {}
    for lang, layer_stats in aggregated.items():
        all_l1 = [v["mean_l1"] for v in layer_stats.values()]
        all_sp = [v["mean_sparsity"] for v in layer_stats.values()]
        summary[lang] = {
            "global_mean_l1": float(np.mean(all_l1)),
            "global_mean_sparsity": float(np.mean(all_sp)),
            "num_layers": len(layer_stats),
        }
        print(
            f"  [{lang}] global_mean_l1={summary[lang]['global_mean_l1']:.4f}  "
            f"global_mean_sparsity={summary[lang]['global_mean_sparsity']:.4f}"
        )

    output = {
        "model": args.model,
        "quantize_kv": args.quantize_kv,
        "num_samples_per_lang": args.num_samples,
        "summary": summary,
        "per_layer": aggregated,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")

    if args.plot:
        plot_results(aggregated, args.output)


if __name__ == "__main__":
    main()
