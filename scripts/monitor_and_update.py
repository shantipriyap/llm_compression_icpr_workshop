#!/usr/bin/env python3
"""
Poll the GPU server every N seconds.
When new result JSON files appear, collect metrics and update results_tables.tex
then push to Overleaf via git.

Usage:
  python3 scripts/monitor_and_update.py

Requires:
  - SSH access to root@95.133.252.51
  - /tmp/ol_check is a git clone of the Overleaf project
  - paramiko:  pip install paramiko
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────────────
SERVER        = "root@95.133.252.51"
REMOTE_DIR    = "/root/icpr/results"
OVERLEAF_DIR  = "/tmp/ol_check/ICPR_2026_LaTeX_Templates"
TABLES_FILE   = f"{OVERLEAF_DIR}/results_tables.tex"
POLL_INTERVAL = 120   # seconds between polls
SSH_OPTS      = "-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# Map result path prefixes → display names used in LaTeX
MODEL_KEYS = {
    "llama3_8b":  "Llama-3.1-8B",
    "gemma3_12b": "Gemma-3-12B",
    "qwen3_8b":   "Qwen3-8B",
    "phi4_mini":  "Phi-4-Mini",
}

COMP_KEYS = {
    "baseline":    "BF16 Baseline",
    "kv_compress": "KV-Quant 4-bit",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def ssh(cmd: str) -> str:
    result = subprocess.run(
        f"ssh {SSH_OPTS} {SERVER} '{cmd}'",
        shell=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def git_cmd(cmd: str, cwd: str = "/tmp/ol_check") -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return (result.stdout + result.stderr).strip()


def fetch_results() -> dict:
    """SSH to server, read all result JSONs, return nested dict."""
    raw = ssh(f"""python3 -c "
import json, glob
out = {{}}
for p in sorted(glob.glob('{REMOTE_DIR}/**/*.json', recursive=True)):
    parts = p.replace('{REMOTE_DIR}/','').split('/')
    if len(parts) < 3: continue
    model, comp, fname = parts[0], parts[1], parts[2]
    bench = fname.replace('_results.json','')
    if bench in ('summary',): continue
    d = json.load(open(p))
    out.setdefault(model, {{}}).setdefault(comp, {{}})[bench] = d
print(json.dumps(out))
" """)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def metric(d: dict) -> str:
    """Extract primary metric value from a result dict."""
    if "accuracy" in d:
        return f"{d['accuracy']*100:.1f}"
    if "rouge_l" in d:
        return f"{d['rouge_l']:.4f}"
    if "chrf" in d:
        return f"{d['chrf']:.1f}"
    if "f1" in d:
        return f"{d['f1']*100:.1f}"
    return "N/A"


def throughput(d: dict) -> Optional[float]:
    return d.get("tokens_per_sec") or d.get("throughput_tok_per_sec")


def vram_gb(d: dict) -> Optional[float]:
    mb = d.get("gpu_memory_mb") or d.get("vram_used_mb") or d.get("memory_mb")
    return round(mb / 1024, 1) if mb else None


# ── LaTeX helpers ────────────────────────────────────────────────────────────
def safe_val(v, fmt=None):
    if v is None or v == "N/A":
        return "--"
    return fmt % v if fmt else str(v)


def build_en_table(results: dict) -> str:
    """Rebuild Table 1 body rows from collected results."""
    models_order = ["qwen3_8b", "phi4_mini", "llama3_8b", "gemma3_12b"]
    rows = []
    for mkey in models_order:
        if mkey not in results:
            continue
        mname = MODEL_KEYS[mkey]
        mdata = results[mkey]
        bl = mdata.get("baseline", {})
        kv = mdata.get("kv_compress", {})

        def row_vals(d):
            gsm = metric(d.get("gsm8k", {})) if "gsm8k" in d else "--"
            bq  = metric(d.get("boolq", {})) if "boolq" in d else "--"
            ms  = metric(d.get("msmarco", {})) if "msmarco" in d else "--"
            vr  = vram_gb(d.get("gsm8k", d.get("boolq", {})))
            return gsm, bq, ms, vr

        bl_g, bl_b, bl_m, bl_v = row_vals(bl)
        kv_g, kv_b, kv_m, kv_v = row_vals(kv)

        def delta(bl_v, kv_v, is_pct=True):
            try:
                b, k = float(bl_v), float(kv_v)
                d = k - b
                return f"{d:+.1f}" if is_pct else f"{d:+.4f}"
            except Exception:
                return "--"

        d_g = delta(bl_g, kv_g)
        d_b = delta(bl_b, kv_b)
        d_m = delta(bl_m, kv_m, is_pct=False)

        short = {"qwen3_8b": "Qwen3-8B", "phi4_mini": "Phi-4-Mini (3.8B)",
                 "llama3_8b": "Llama-3.1-8B (8B)", "gemma3_12b": "Gemma-3-12B"}[mkey]

        r = f"\\midrule\n\\multirow{{2}}{{*}}{{\\shortstack[l]{{{mname}\\\\({_param(mkey)})}}}}\n"
        vr_str = f"{bl_v}" if bl_v else "--"

        # Bold best GSM8K across all models
        gsm_bl = f"\\textbf{{{bl_g}}}" if mkey == "phi4_mini" else bl_g
        gsm_kv = f"\\textbf{{{kv_g}}}" if mkey == "phi4_mini" else kv_g

        r += f"  & BF16 Baseline  & {gsm_bl} & -- & {bl_b} & -- & {bl_m} & -- & {vr_str} \\\\\n"
        r += f"  & KV-Quant 4-bit & {gsm_kv} & {d_g} & {kv_b} & {d_b} & {kv_m} & {d_m} & {vr_str} \\\\"
        rows.append(r)
    return "\n".join(rows)


def _param(mkey):
    return {"qwen3_8b": "8B", "phi4_mini": "3.8B",
            "llama3_8b": "8B", "gemma3_12b": "12B"}.get(mkey, "?B")


# ── Main monitor loop ────────────────────────────────────────────────────────
def fingerprint(results: dict) -> str:
    """A hash of all result keys to detect new completions."""
    keys = sorted(
        f"{m}/{c}/{b}"
        for m, md in results.items()
        for c, cd in md.items()
        for b in cd
    )
    return "|".join(keys)


def update_overleaf(results: dict, prev_fp: str) -> str:
    new_fp = fingerprint(results)
    if new_fp == prev_fp:
        return prev_fp  # nothing changed

    # Pull latest Overleaf state
    git_cmd("git pull --rebase -q", cwd="/tmp/ol_check")

    with open(TABLES_FILE) as f:
        tex = f.read()

    changed = False

    # ── Update Table 1 rows ──────────────────────────────────────────────────
    for mkey in ["llama3_8b", "gemma3_12b"]:
        if mkey not in results:
            continue
        mdata = results[mkey]
        bl = mdata.get("baseline", {})
        kv = mdata.get("kv_compress", {})
        if not bl.get("gsm8k") or not bl.get("boolq") or not bl.get("msmarco"):
            continue
        if not kv.get("gsm8k") or not kv.get("boolq") or not kv.get("msmarco"):
            continue

        mname    = MODEL_KEYS[mkey]
        param    = _param(mkey)
        bl_g = metric(bl["gsm8k"]); bl_b = metric(bl["boolq"]); bl_m = metric(bl["msmarco"])
        kv_g = metric(kv["gsm8k"]); kv_b = metric(kv["boolq"]); kv_m = metric(kv["msmarco"])
        vr   = vram_gb(bl["gsm8k"]) or "--"

        def delta_pp(a, b):
            try: return f"{float(b)-float(a):+.1f}"
            except: return "--"
        def delta_rl(a, b):
            try: return f"{float(b)-float(a):+.4f}"
            except: return "--"

        new_rows = (
            f"\\midrule\n"
            f"\\multirow{{2}}{{*}}{{\\shortstack[l]{{{mname}\\\\({param}B)}}}}\n"
            f"  & BF16 Baseline  & {bl_g} & --   & {bl_b} & --   & {bl_m} & --      & {vr} \\\\\n"
            f"  & KV-Quant 4-bit & {kv_g} & {delta_pp(bl_g,kv_g)} & {kv_b} & {delta_pp(bl_b,kv_b)} & {kv_m} & {delta_rl(bl_m,kv_m)} & {vr} \\\\"
        )

        # Check if already in the table
        if mname in tex and "BF16 Baseline" in tex.split(mname)[1][:400]:
            print(f"  {mkey} already in table, skipping")
            continue

        # Insert before \bottomrule in Table 1 (tab:main_results)
        tex = tex.replace(
            "\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\n  \\small\n  \\item[a] KV-Quant:",
            new_rows + "\n\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\n  \\small\n  \\item[a] KV-Quant:"
        )
        changed = True
        print(f"  ✓ Added {mkey} to Table 1")

    # ── Update Table 2 throughput rows ────────────────────────────────────────
    for mkey in ["llama3_8b", "gemma3_12b"]:
        if mkey not in results:
            continue
        mdata = results[mkey]
        for comp in ["baseline", "kv_compress"]:
            d = mdata.get(comp, {})
            gsm = d.get("gsm8k", {}); bq = d.get("boolq", {}); ms = d.get("msmarco", {})
            if not gsm or not bq or not ms:
                continue
            t_g = throughput(gsm); t_b = throughput(bq); t_m = throughput(ms)
            if not t_g:
                continue
            mname = MODEL_KEYS[mkey]; param = _param(mkey)
            comp_label = COMP_KEYS[comp]
            row = f"  & {comp_label}  & {t_g:.1f} & {t_b:.1f} & {t_m:.1f} \\\\"
            tag = f"{mname}\\\\({param}B)"
            if tag in tex and comp_label in tex.split(tag)[1][:300]:
                continue
            if tag not in tex:
                tex = tex.replace(
                    "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\n% ---- Table 3:",
                    f"\\midrule\n\\multirow{{2}}{{*}}{{\\shortstack[l]{{{mname}\\\\({param}B)}}}}\n{row}\n"
                    + "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\n% ---- Table 3:"
                )
                changed = True
                print(f"  ✓ Added {mkey}/{comp} to Table 2")

    if changed:
        with open(TABLES_FILE, "w") as f:
            f.write(tex)
        msg = "Auto-update: " + ", ".join(
            f"{m} {c}" for m in ["llama3_8b","gemma3_12b"]
            for c in ["baseline","kv_compress"]
            if results.get(m, {}).get(c, {}).get("gsm8k")
        )
        out = git_cmd(f'git add ICPR_2026_LaTeX_Templates/results_tables.tex && git commit -m "{msg}" && git push origin master',
                      cwd="/tmp/ol_check")
        print(f"  → Pushed to Overleaf: {out.splitlines()[-1] if out else 'OK'}")
    else:
        print("  No new complete result sets yet.")

    return new_fp


def completed_summary(results: dict) -> str:
    lines = []
    for mkey in ["qwen3_8b", "phi4_mini", "llama3_8b", "gemma3_12b"]:
        mdata = results.get(mkey, {})
        for comp in ["baseline", "kv_compress"]:
            d = mdata.get(comp, {})
            benches = [b for b in ["gsm8k","boolq","msmarco"] if b in d]
            if benches:
                vals = [f"{b}={metric(d[b])}" for b in benches]
                lines.append(f"  {mkey}/{comp}: {', '.join(vals)}")
    return "\n".join(lines) if lines else "  (none)"


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ICPR 2026 — Monitor & Auto-Update Overleaf")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print("=" * 60)

    prev_fp = ""
    iteration = 0

    while True:
        iteration += 1
        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}] Poll #{iteration}")

        results = fetch_results()

        print("  Completed results:")
        print(completed_summary(results))

        # GPU status
        gpu = ssh("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null")
        if gpu:
            parts = [p.strip() for p in gpu.split(",")]
            print(f"  GPU: {parts[0]}% util · {int(parts[1])//1024}GB used · {parts[2]}°C")

        # Current benchmark progress
        progress = ssh("tmux capture-pane -t icpr_missing -p -S -5 2>/dev/null | grep -E 'it/s|it\]' | tail -2")
        if progress:
            print(f"  Active: {progress.strip()}")

        prev_fp = update_overleaf(results, prev_fp)

        # Check if all done
        done_keys = set(fingerprint(results).split("|"))
        all_target = {
            "llama3_8b/baseline/gsm8k","llama3_8b/baseline/boolq","llama3_8b/baseline/msmarco",
            "llama3_8b/kv_compress/gsm8k","llama3_8b/kv_compress/boolq","llama3_8b/kv_compress/msmarco",
            "gemma3_12b/baseline/gsm8k","gemma3_12b/baseline/boolq","gemma3_12b/baseline/msmarco",
            "gemma3_12b/kv_compress/gsm8k","gemma3_12b/kv_compress/boolq","gemma3_12b/kv_compress/msmarco",
        }
        missing = all_target - done_keys
        if not missing:
            print("\n✅ All target experiments complete! Overleaf up to date.")
            break

        remaining = len(missing)
        print(f"  Still waiting for {remaining} result files...")
        time.sleep(POLL_INTERVAL)
