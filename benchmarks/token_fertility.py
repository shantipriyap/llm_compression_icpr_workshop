"""
token_fertility.py — Measures tokenizer fertility per language.

Fertility = average number of tokens produced per whitespace-delimited word.
A value > 1 means the tokenizer is less efficient for that language, directly
increasing KV-cache size under compression and explaining why KV-cache
quantization has a larger impact on Indic languages.

Compression relevance:
  If Odia input produces 3× more tokens than English, a KV-cache quantization
  policy that discards the bottom-k% of attention positions will discard
  proportionally more Indic-language context, explaining our observed
  instruction-following failures.

Usage:
  python benchmarks/token_fertility.py --models Qwen/Qwen3-8B microsoft/Phi-4-mini-instruct
  python benchmarks/token_fertility.py --models Qwen/Qwen3-8B --output results/fertility.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

# ── Test sentences (same semantic content in each language) ───────────────────

# Five models spanning 0.6B → 8B; used as default when --models not specified
# All are non-gated; TinyLlama provides Llama-family tokenizer without gating
DEFAULT_MODELS = [
    "Qwen/Qwen3-0.6B",                # 0.6 B  (Qwen3 tokenizer)
    "Qwen/Qwen3-1.7B",                # 1.7 B  (Qwen3 tokenizer)
    "microsoft/Phi-4-mini-instruct",  # 3.8 B  (Phi tokenizer)
    "Qwen/Qwen3-8B",                  # 8 B    (Qwen3 tokenizer)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1 B (Llama tokenizer family)
]

SENTENCES: Dict[str, List[str]] = {
    "en": [
        "What is the capital of France?",
        "Machine learning is a subset of artificial intelligence.",
        "The water cycle describes how water evaporates from surfaces.",
        "Democracy is a system of government by the whole population.",
        "Large language models are trained on massive text corpora.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The speed of light in a vacuum is approximately 299,792 km/s.",
        "Neural networks are inspired by the structure of the human brain.",
    ],
    # ── Indo-Aryan / Devanagari ───────────────────────────────────────────────
    "hi": [
        "फ्रांस की राजधानी पेरिस है।",
        "मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक उपसमुच्चय है।",
        "जल चक्र बताता है कि पानी सतहों से कैसे वाष्पित होता है।",
        "लोकतंत्र पूरी जनसंख्या द्वारा सरकार की एक प्रणाली है।",
        "बड़े भाषा मॉडलों को विशाल पाठ कोष पर प्रशिक्षित किया जाता है।",
        "प्रकाश संश्लेषण सूर्य के प्रकाश को रासायनिक ऊर्जा में बदलता है।",
        "निर्वात में प्रकाश की गति लगभग 299,792 किमी/से है।",
        "न्यूरल नेटवर्क मानव मस्तिष्क की संरचना से प्रेरित हैं।",
    ],
    "mr": [
        "फ्रान्सची राजधानी पॅरिस आहे.",
        "मशीन लर्निंग हे कृत्रिम बुद्धिमत्तेचा एक उपसंच आहे.",
        "जलचक्र पाणी पृष्ठभागावरून कसे बाष्प होते हे सांगते.",
        "लोकशाही म्हणजे संपूर्ण लोकसंख्येद्वारे सरकार चालवण्याची व्यवस्था.",
        "मोठे भाषा मॉडेल्स प्रचंड मजकूर संग्रहांवर प्रशिक्षित केले जातात.",
        "प्रकाश संश्लेषण सूर्यप्रकाशाचे रासायनिक ऊर्जेत रूपांतर करते.",
        "निर्वातात प्रकाशाचा वेग सुमारे 299,792 किमी/से आहे.",
        "न्यूरल नेटवर्क्स मानवी मेंदूच्या संरचनेतून प्रेरित आहेत.",
    ],
    "gu": [
        "ફ્રાન્સની રાજધાની પેરિસ છે.",
        "મશીન લર્નિંગ એ કૃત્રિમ બુદ્ધિમત્તાનો ઉપગણ છે.",
        "જળ ચક્ર વર્ણવે છે કે સપાટી પરથી પાણી કેવી રીતે બાષ્પ થાય છે.",
        "લોકશાહી એ સમગ્ર વસ્તી દ્વારા સરકારની વ્યવસ્થા છે.",
        "મોટા ભાષા મૉડલ્સ વિશાળ ટેક્સ્ટ સંગ્રહ પર પ્રશિક્ષિત છે.",
        "પ્રકાશ સંશ્લેષણ સૂર્યપ્રકાશને રાસાયણિક ઊર્જામાં ફેરવે છે.",
        "નિર્વાતમાં પ્રકાશની ઝડપ લગભગ 299,792 કિ.મી./સે. છે.",
        "ન્યુરલ નેટવર્ક્સ માનવ મગજની રચનાથી પ્રેરિત છે.",
    ],
    # ── Indo-Aryan / own scripts ──────────────────────────────────────────────
    "or": [
        "ଫ୍ରାନ୍ସର ରାଜଧାନୀ ପ୍ୟାରିସ ଅଟେ।",
        "ମେସିନ ଲର୍ନିଂ ହେଉଛି କୃତ୍ରିମ ବୁଦ୍ଧିମତ୍ତାର ଏକ ଉପ-ଗଣ।",
        "ଜଳ ଚକ୍ର ବର୍ଣ୍ଣନା କରେ ଯେ ଜଳ ସ୍ତରରୁ କିପରି ବাஷ୍ପ ହୁଏ।",
        "ଗଣତନ୍ତ୍ର ହେଉଛି ସମଗ୍ର ଜନଗଣ ଦ୍ୱାରା ସରକାର ବ୍ୟବସ୍ଥା।",
        "ବୃହତ ଭାଷା ମଡେଲଗୁଡ଼ିକ ବିଶାଳ ପାଠ୍ୟ ସଂଗ୍ରହ ଉପରେ ତାଲିମ ପ୍ରାପ୍ତ।",
        "ଆଲୋକ ସଂଶ୍ଳେଷଣ ସୂର୍ଯ୍ୟ ଆଲୋକକୁ ରାସାୟନିକ ଶକ୍ତିରେ ରୂପାନ୍ତର କରେ।",
        "ଶୂନ୍ୟ ସ୍ଥାନରେ ଆଲୋକର ବେଗ ପ୍ରାୟ 299,792 କି.ମି./ସେ.।",
        "ନ୍ୟୁରାଲ ନେଟୱାର୍କ ମାନବ ମସ୍ତିଷ୍କ ରଚନାରୁ ଅନୁପ୍ରାଣିତ।",
    ],
    "bn": [
        "ফ্রান্সের রাজধানী প্যারিস।",
        "মেশিন লার্নিং কৃত্রিম বুদ্ধিমত্তার একটি উপসেট।",
        "জলচক্র বর্ণনা করে কীভাবে পৃষ্ঠ থেকে জল বাষ্পীভূত হয়।",
        "গণতন্ত্র হল সমগ্র জনগণের দ্বারা সরকার ব্যবস্থা।",
        "বড় ভাষা মডেলগুলি বিশাল পাঠ্য কর্পাসে প্রশিক্ষিত।",
        "সালোকসংশ্লেষণ সূর্যালোককে রাসায়নিক শক্তিতে রূপান্তরিত করে।",
        "শূন্যে আলোর গতি প্রায় 299,792 কি.মি./সে.।",
        "নিউরাল নেটওয়ার্ক মানব মস্তিষ্কের গঠন থেকে অনুপ্রাণিত।",
    ],
    # ── Dravidian ─────────────────────────────────────────────────────────────
    "ta": [
        "பிரான்சின் தலைநகரம் பாரிஸ் ஆகும்.",
        "இயந்திர கற்றல் செயற்கை நுண்ணறிவின் ஒரு உட்பிரிவு ஆகும்.",
        "நீர் சுழற்சி நீர் மேற்பரப்பிலிருந்து எவ்வாறு ஆவியாகிறது என்பதை விவரிக்கிறது.",
        "ஜனநாயகம் என்பது மக்கள் அனைவரும் ஆளும் ஒரு அரசு முறை.",
        "பெரிய மொழி மாதிரிகள் பெரும் உரை தரவுகளில் பயிற்சி பெறுகின்றன.",
        "ஒளிச்சேர்க்கை சூரிய வெளிச்சத்தை இரசாயன ஆற்றலாக மாற்றுகிறது.",
        "வெற்றிடத்தில் ஒளியின் வேகம் தோராயமாக 299,792 கி.மீ./வி.",
        "நரம்பியல் வலைப்பின்னல்கள் மனித மூளையின் கட்டமைப்பால் தூண்டப்படுகின்றன.",
    ],
    "te": [
        "ఫ్రాన్స్ రాజధాని పారిస్.",
        "మెషిన్ లెర్నింగ్ కృత్రిమ మేధస్సులో ఒక ఉపసమితి.",
        "నీటి చక్రం ఉపరితలాల నుండి నీరు ఎలా ఆవిరవుతుందో వివరిస్తుంది.",
        "ప్రజాస్వామ్యం అనేది మొత్తం జనాభాచేత ప్రభుత్వ వ్యవస్థ.",
        "పెద్ద భాషా నమూనాలు విస్తారమైన వ్యాసాల సంకలనాలపై శిక్షణ పొందుతాయి.",
        "కిరణజన్య సంయోగక్రియ సూర్యకాంతిని రసాయన శక్తిగా మారుస్తుంది.",
        "శూన్యంలో కాంతి వేగం సుమారు 299,792 కి.మీ./సె.",
        "న్యూరల్ నెట్‍వర్క్‍లు మానవ మెదడు నిర్మాణం ద్వారా ప్రేరణ పొందాయి.",
    ],
    "kn": [
        "ಫ್ರಾನ್ಸ್‌ನ ರಾಜಧಾನಿ ಪ್ಯಾರಿಸ್.",
        "ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆಯ ಒಂದು ಉಪಸಮೂಹ.",
        "ನೀರಿನ ಚಕ್ರವು ನೀರು ಮೇಲ್ಮೈಗಳಿಂದ ಹೇಗೆ ಆವಿಯಾಗುತ್ತದೆ ಎಂಬುದನ್ನು ವಿವರಿಸುತ್ತದೆ.",
        "ಪ್ರಜಾಪ್ರಭುತ್ವ ಎಂದರೆ ಇಡೀ ಜನಸಂಖ್ಯೆಯ ಆಡಳಿತ ವ್ಯವಸ್ಥೆ.",
        "ದೊಡ್ಡ ಭಾಷಾ ಮಾದರಿಗಳು ಅಗಾಧ ಪಠ್ಯ ಸಂಗ್ರಹಗಳ ಮೇಲೆ ತರಬೇತಿ ಪಡೆಯುತ್ತವೆ.",
        "ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ ಸೂರ್ಯಪ್ರಕಾಶವನ್ನು ರಾಸಾಯನಿಕ ಶಕ್ತಿಯಾಗಿ ಪರಿವರ್ತಿಸುತ್ತದೆ.",
        "ನಿರ್ವಾತದಲ್ಲಿ ಬೆಳಕಿನ ವೇಗ ಸುಮಾರು 299,792 ಕಿ.ಮೀ./ಸೆ.",
        "ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್‌ಗಳು ಮಾನವ ಮೆದುಳಿನ ರಚನೆಯಿಂದ ಸ್ಫೂರ್ತಿ ಪಡೆದಿವೆ.",
    ],
    "ml": [
        "ഫ്രാൻസിന്റെ തലസ്ഥാനം പാരീസ് ആണ്.",
        "മെഷീൻ ലേണിംഗ് കൃത്രിമ ബുദ്ധിയുടെ ഒരു ഉപഗണമാണ്.",
        "ജലചക്രം ഉപരിതലങ്ങളിൽ നിന്ന് ജലം എങ്ങനെ ബാഷ്പീകരിക്കുന്നു എന്ന് വിവരിക്കുന്നു.",
        "ജനാധിപത്യം എന്നത് മുഴുവൻ ജനങ്ങളുടെ ഭരണ സംവിധാനമാണ്.",
        "വലിയ ഭാഷാ മോഡലുകൾ വിശാലമായ ടെക്സ്റ്റ് ശേഖരങ്ങളിൽ പരിശീലിപ്പിക്കപ്പെടുന്നു.",
        "ഫോട്ടോസിന്തസിസ് സൂര്യപ്രകാശത്തെ രാസ ഊർജ്ജമാക്കി മാറ്റുന്നു.",
        "ശൂന്യതയിൽ പ്രകാശത്തിന്റെ വേഗത ഏകദേശം 299,792 കി.മീ./സെ.",
        "ന്യൂറൽ നെറ്റ്‌വർക്കുകൾ മനുഷ്യ മസ്തിഷ്കത്തിന്റെ ഘടനയിൽ നിന്ന് പ്രചോദനം ഉൾക്കൊള്ളുന്നു.",
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


def compute_fertility(tokenizer, sentences: List[str]) -> Dict:
    """
    Returns:
        mean_fertility: tokens / whitespace-word (main metric)
        mean_tokens:    mean token count per sentence
        mean_words:     mean word count per sentence
    """
    total_tokens = 0
    total_words = 0
    for sent in sentences:
        toks = tokenizer.encode(sent, add_special_tokens=False)
        words = sent.split()
        total_tokens += len(toks)
        total_words += len(words)
    return {
        "mean_fertility": round(total_tokens / total_words, 3) if total_words else 0,
        "mean_tokens_per_sentence": round(total_tokens / len(sentences), 1),
        "mean_words_per_sentence": round(total_words / len(sentences), 1),
    }


def print_table(results: Dict[str, Dict[str, Dict]]):
    """Pretty-print an ASCII table: rows=languages, cols=models."""
    models = list(results.keys())
    langs = list(LANG_NAMES.keys())
    col_w = 18

    header = f"{'Language':<12}" + "".join(f"{m[-16:]:<{col_w}}" for m in models)
    print("\n=== Token Fertility (tokens / word) ===")
    print(header)
    print("-" * len(header))
    for lang in langs:
        row = f"{LANG_NAMES[lang]:<12}"
        for m in models:
            f_val = results[m].get(lang, {}).get("mean_fertility", "N/A")
            row += f"{f_val:<{col_w}}"
        print(row)

    print("\nHigher fertility = more tokens per word = more KV-cache pressure under compression.\n")


def main():
    parser = argparse.ArgumentParser(description="Token fertility analysis across languages")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HF model IDs to compare (tokenizer only; no GPU needed). "
             "Defaults to 5 models spanning 1B–70B."
    )
    parser.add_argument(
        "--languages", nargs="+", default=list(LANG_NAMES.keys()),
        choices=list(LANG_NAMES.keys())
    )
    parser.add_argument("--output", default="results/token_fertility.json")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    results: Dict[str, Dict] = {}

    for model_id in args.models:
        print(f"\nLoading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        results[model_id] = {}
        for lang in args.languages:
            stats = compute_fertility(tokenizer, SENTENCES[lang])
            results[model_id][lang] = stats
            print(
                f"  [{LANG_NAMES[lang]:>8}] fertility={stats['mean_fertility']:.3f}  "
                f"tokens/sent={stats['mean_tokens_per_sentence']:.1f}  "
                f"words/sent={stats['mean_words_per_sentence']:.1f}"
            )

    print_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
