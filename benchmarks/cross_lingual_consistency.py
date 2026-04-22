"""
cross_lingual_consistency.py — Measures compression-induced semantic drift across languages.

Consistency = whether a compressed model produces the same answer as the
baseline model when asked the same factual question in different languages.
A fully consistent model should answer "Paris" to "capital of France?" in
all languages, regardless of compression.

Compression relevance:
  If KV-Quant 4-bit consistency(EN)=0.90 and consistency(OR)=0.15, the metric
  proves that compression specifically degrades Odia responses beyond its effect
  on English — quantifying multilingual compression disparity as a single number.

Metric:
  Consistency score per (model, compression, language) = fraction of questions
  where the compressed model's answer semantically matches the baseline answer
  for the same question in that language. Matching is soft: checks if the
  expected answer token appears in the first 50 generated tokens.

Usage:
  python benchmarks/cross_lingual_consistency.py \\
      --model Qwen/Qwen3-8B \\
      --baseline_method baseline \\
      --compressed_method kv_compress \\
      --output results/consistency.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Factual QA pairs: (question, expected_answer_substring) ──────────────────
# Same question in EN/HI/OR/BN; expected answer is language-agnostic string
# that should appear in model output for ANY language version.

QA_PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "en": [
        ("What is the capital of France?", "paris"),
        ("What is the chemical formula for water?", "h2o"),
        ("Who wrote Romeo and Juliet?", "shakespeare"),
        ("What planet is closest to the Sun?", "mercury"),
        ("How many continents are on Earth?", "seven"),
        ("What is the largest ocean?", "pacific"),
        ("What gas do plants absorb?", "carbon"),
        ("What is 12 multiplied by 12?", "144"),
    ],
    # ── Indo-Aryan / Devanagari ───────────────────────────────────────────────
    "hi": [
        ("फ्रांस की राजधानी क्या है?", "paris"),
        ("पानी का रासायनिक सूत्र क्या है?", "h2o"),
        ("रोमियो और जूलियट किसने लिखा?", "shakespeare"),
        ("सूर्य के सबसे निकट कौन सा ग्रह है?", "mercury"),
        ("पृथ्वी पर कितने महाद्वीप हैं?", "seven"),
        ("सबसे बड़ा महासागर कौन सा है?", "pacific"),
        ("पौधे कौन सी गैस अवशोषित करते हैं?", "carbon"),
        ("12 गुणा 12 कितना होता है?", "144"),
    ],
    "mr": [
        ("फ्रान्सची राजधानी काय आहे?", "paris"),
        ("पाण्याचे रासायनिक सूत्र काय आहे?", "h2o"),
        ("रोमियो आणि ज्युलिएट कोणी लिहिले?", "shakespeare"),
        ("सूर्याच्या सर्वात जवळचा ग्रह कोणता?", "mercury"),
        ("पृथ्वीवर किती खंड आहेत?", "seven"),
        ("सर्वात मोठा महासागर कोणता?", "pacific"),
        ("झाडे कोणता वायू शोषतात?", "carbon"),
        ("12 गुणिले 12 किती?", "144"),
    ],
    "gu": [
        ("ફ્રાન્સની રાજધાની કઈ છે?", "paris"),
        ("પાણીનું રાસાયણિક સૂત્ર શું છે?", "h2o"),
        ("રોમિયો અને જ્યુલિયટ કોણે લખ્યું?", "shakespeare"),
        ("સૂર્યની સૌથી નજીકનો ગ્રહ કયો છે?", "mercury"),
        ("પૃથ્વી પર કેટલા ખંડ છે?", "seven"),
        ("સૌથી મોટો મહાસાગર કયો છે?", "pacific"),
        ("છોડ કઈ ગેસ શોષે છે?", "carbon"),
        ("12 ગુણ્યા 12 કેટલા?", "144"),
    ],
    # ── Indo-Aryan / own scripts ──────────────────────────────────────────────
    "or": [
        ("ଫ୍ରାନ୍ସର ରାଜଧାନୀ କ'ଣ?", "paris"),
        ("ଜଳର ରାସାୟନିକ ସୂତ୍ର କ'ଣ?", "h2o"),
        ("ରୋମିଓ ଓ ଜୁଲିଏଟ୍ କିଏ ଲେଖିଥିଲେ?", "shakespeare"),
        ("ସୂର୍ଯ୍ୟ ନିକଟସ୍ଥ ଗ୍ରହ କ'ଣ?", "mercury"),
        ("ପୃଥିବୀରେ କେତୋଟି ମହାଦ୍ୱୀପ ଅଛି?", "seven"),
        ("ସବୁଠୁ ବଡ଼ ମହାସାଗର କ'ଣ?", "pacific"),
        ("ଗଛ କେଉଁ ଗ୍ୟାସ ଶୋଷଣ କରେ?", "carbon"),
        ("12 ଗୁଣ 12 କେତେ?", "144"),
    ],
    "bn": [
        ("ফ্রান্সের রাজধানী কী?", "paris"),
        ("পানির রাসায়নিক সূত্র কী?", "h2o"),
        ("রোমিও ও জুলিয়েট কে লিখেছেন?", "shakespeare"),
        ("সূর্যের সবচেয়ে কাছের গ্রহ কোনটি?", "mercury"),
        ("পৃথিবীতে কটি মহাদেশ আছে?", "seven"),
        ("সবচেয়ে বড় মহাসাগর কোনটি?", "pacific"),
        ("গাছ কোন গ্যাস শোষণ করে?", "carbon"),
        ("12 গুণ 12 কত?", "144"),
    ],
    # ── Dravidian ─────────────────────────────────────────────────────────────
    "ta": [
        ("பிரான்சின் தலைநகரம் என்ன?", "paris"),
        ("நீரின் வேதியியல் சூத்திரம் என்ன?", "h2o"),
        ("ரோமியோ மற்றும் ஜூலியட்டை யார் எழுதினார்?", "shakespeare"),
        ("சூர்யனுக்கு மிக அருகில் உள்ள கிரகம் எது?", "mercury"),
        ("பூமியில் எத்தனை கண்டங்கள் உள்ளன?", "seven"),
        ("மிகப் பெரிய மகாசமுத்திரம் எது?", "pacific"),
        ("தாவரங்கள் எந்த வாயுவை உறிஞ்சுகின்றன?", "carbon"),
        ("12 பெருக்கல் 12 என்ன?", "144"),
    ],
    "te": [
        ("ఫ్రాన్స్ రాజధాని ఏమిటి?", "paris"),
        ("నీటి రసాయన సూత్రం ఏమిటి?", "h2o"),
        ("రోమియో మరియు జూలియట్‌ను ఎవరు రాశారు?", "shakespeare"),
        ("సూర్యుడికి దగ్గరగా ఉన్న గ్రహం ఏది?", "mercury"),
        ("భూమిపై ఎన్ని ఖండాలు ఉన్నాయి?", "seven"),
        ("అతిపెద్ద మహాసముద్రం ఏది?", "pacific"),
        ("మొక్కలు ఏ వాయువును గ్రహిస్తాయి?", "carbon"),
        ("12 గుణకారం 12 ఎంత?", "144"),
    ],
    "kn": [
        ("ಫ್ರಾನ್ಸ್‌ನ ರಾಜಧಾನಿ ಏನು?", "paris"),
        ("ನೀರಿನ ರಾಸಾಯನಿಕ ಸೂತ್ರ ಏನು?", "h2o"),
        ("ರೋಮಿಯೋ ಮತ್ತು ಜೂಲಿಯೆಟ್ ಅನ್ನು ಯಾರು ಬರೆದರು?", "shakespeare"),
        ("ಸೂರ್ಯನಿಗೆ ಹತ್ತಿರದ ಗ್ರಹ ಯಾವುದು?", "mercury"),
        ("ಭೂಮಿಯಲ್ಲಿ ಎಷ್ಟು ಖಂಡಗಳಿವೆ?", "seven"),
        ("ಅತಿದೊಡ್ಡ ಮಹಾಸಾಗರ ಯಾವುದು?", "pacific"),
        ("ಸಸ್ಯಗಳು ಯಾವ ಅನಿಲವನ್ನು ಹೀರಿಕೊಳ್ಳುತ್ತವೆ?", "carbon"),
        ("12 ಗುಣಿತ 12 ಎಷ್ಟು?", "144"),
    ],
    "ml": [
        ("ഫ്രാൻസിന്റെ തലസ്ഥാനം ഏതാണ്?", "paris"),
        ("ജലത്തിന്റെ രാസ സൂത്രവാക്യം എന്താണ്?", "h2o"),
        ("റോമിയോ ആൻഡ് ജൂലിയറ്റ് ആരെഴുതി?", "shakespeare"),
        ("സൂര്യനോട് ഏറ്റവും അടുത്ത ഗ്രഹം ഏതാണ്?", "mercury"),
        ("ഭൂമിയിൽ എത്ര ഭൂഖണ്ഡങ്ങൾ ഉണ്ട്?", "seven"),
        ("ഏറ്റവും വലിയ സമുദ്രം ഏതാണ്?", "pacific"),
        ("സസ്യങ്ങൾ ഏത് വാതകം ആഗിരണം ചെയ്യുന്നു?", "carbon"),
        ("12 ഗുണം 12 എത്ര?", "144"),
    ],
}

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi (Devanagari)",
    "mr": "Marathi (Devanagari)",
    "gu": "Gujarati",
    "or": "Odia",
    "bn": "Bengali",
    "ta": "Tamil (Dravidian)",
    "te": "Telugu (Dravidian)",
    "kn": "Kannada (Dravidian)",
    "ml": "Malayalam (Dravidian)",
}


def load_model(model_id: str, use_4bit_nf4: bool, hf_token: str | None):
    kwargs = dict(trust_remote_code=True, token=hf_token, device_map="auto")
    if use_4bit_nf4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 50) -> str:
    device = next(model.parameters()).device
    # Use chat template with thinking disabled (Qwen3 enable_thinking=False)
    # so the model gives direct answers rather than lengthy CoT blocks.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": question}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
        except TypeError:
            enc = tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
    else:
        enc = tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc.input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    # Fallback: extract after </think> in case thinking was still emitted
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def score_consistency(
    model,
    tokenizer,
    languages: List[str],
    max_new_tokens: int = 50,
) -> Dict[str, float]:
    """
    Returns fraction of questions where the expected answer substring appears
    in the generated output, per language.
    """
    scores: Dict[str, float] = {}
    for lang in languages:
        hits = 0
        pairs = QA_PAIRS[lang]
        for question, expected in pairs:
            answer = generate_answer(model, tokenizer, question, max_new_tokens)
            if expected in answer:
                hits += 1
            else:
                # Tolerance: some models answer in the input language script
                # The expected token is always a Latin/digit string (paris, h2o, 144...)
                pass
        scores[lang] = round(hits / len(pairs), 3)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Cross-lingual consistency under compression"
    )
    parser.add_argument("--model", required=True, help="HF model ID")
    parser.add_argument(
        "--languages", nargs="+", default=list(LANG_NAMES.keys()),
        choices=list(LANG_NAMES.keys()),
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50,
        help="Tokens to generate per question",
    )
    parser.add_argument("--output", default="results/consistency.json")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=hf_token
    )

    results = {"model": args.model, "variants": {}}

    for label, use_4bit in [("baseline_bf16", False), ("compressed_nf4", True)]:
        print(f"\n--- Loading {label} ---")
        model = load_model(args.model, use_4bit_nf4=use_4bit, hf_token=hf_token)
        scores = score_consistency(model, tokenizer, args.languages, args.max_new_tokens)
        results["variants"][label] = scores

        print(f"  Consistency scores ({label}):")
        for lang, score in scores.items():
            bar = "█" * int(score * 20)
            print(f"    {LANG_NAMES[lang]:>8}: {score:.3f}  {bar}")

        # Free VRAM before next model load
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute delta: how much compression hurts each language
    baseline = results["variants"].get("baseline_bf16", {})
    compressed = results["variants"].get("compressed_nf4", {})
    results["compression_delta"] = {
        lang: round(compressed.get(lang, 0) - baseline.get(lang, 0), 3)
        for lang in args.languages
    }

    print("\n=== Compression Delta (compressed − baseline) ===")
    for lang, delta in results["compression_delta"].items():
        sign = "+" if delta >= 0 else ""
        print(f"  {LANG_NAMES[lang]:>8}: {sign}{delta:.3f}")

    print("\nNegative delta = compression degrades that language.")
    print("Larger negative delta for Indic vs EN → disproportionate compression harm.\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
