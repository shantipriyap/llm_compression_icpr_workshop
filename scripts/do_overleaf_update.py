#!/usr/bin/env python3
"""
Poll GPU server, collect results, update results_tables.tex, push to Overleaf.
Run once or in a loop.
"""
import json, subprocess, sys, time

SSH = "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@95.133.252.51"
OL  = "/tmp/ol_check"
TEX = f"{OL}/ICPR_2026_LaTeX_Templates/results_tables.tex"

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return (r.stdout + r.stderr).strip()

def ssh(cmd):
    return run(f'{SSH} "{cmd}"')

# ─── Fetch all results from server ───────────────────────────────────────────
def fetch():
    script = r"""
import json,glob
out={}
for p in sorted(glob.glob('/root/icpr/results/**/*.json',recursive=True)):
    parts=p.replace('/root/icpr/results/','').split('/')
    if len(parts)<3: continue
    model,comp,fname=parts[0],parts[1],parts[2]
    bench=fname.replace('_results.json','')
    if bench=='summary': continue
    try: d=json.load(open(p))
    except: continue
    out.setdefault(model,{}).setdefault(comp,{})[bench]={k:round(v,5) if isinstance(v,float) else v for k,v in d.items() if k in ['accuracy','rouge_l','tokens_per_sec','gpu_memory_mb','num_samples']}
print(json.dumps(out))
"""
    raw = ssh(f"python3 -c '{script}'")
    try:
        return json.loads(raw)
    except:
        return {}

def m(d, key):
    """Primary metric from a benchmark result dict."""
    if not d: return None
    if 'accuracy' in d: return round(d['accuracy']*100, 1)
    if 'rouge_l'  in d: return round(d['rouge_l'], 4)
    return None

def fmt(v):
    return str(v) if v is not None else '--'

def delta(a, b):
    if a is None or b is None: return '--'
    d = round(b - a, 1)
    return f"{d:+.1f}"

def delta_rl(a, b):
    if a is None or b is None: return '--'
    d = round(b - a, 4)
    return f"{d:+.4f}"

def tput(d):
    v = d.get('tokens_per_sec')
    return round(v, 1) if v else None

def vram(d):
    v = d.get('gpu_memory_mb')
    return round(v/1024, 1) if v else None

MODELS = {
    'llama3_8b':  ('Llama-3.1-8B', '8'),
    'gemma3_12b': ('Gemma-3-12B',  '12'),
}

# ─── Build new LaTeX rows ─────────────────────────────────────────────────────
def make_table1_rows(results, existing_tex):
    rows = []
    for mkey, (mname, param) in MODELS.items():
        # Skip if already in the table
        if mname in existing_tex:
            continue
        mdata = results.get(mkey, {})
        bl = mdata.get('baseline', {})
        kv = mdata.get('kv_compress', {})
        # Need all three EN benchmarks for both methods to add a complete block
        if not (bl.get('gsm8k') and bl.get('boolq') and bl.get('msmarco')):
            continue
        if not (kv.get('gsm8k') and kv.get('boolq') and kv.get('msmarco')):
            continue

        bl_g = m(bl['gsm8k']); bl_b = m(bl['boolq']); bl_r = m(bl['msmarco'])
        kv_g = m(kv['gsm8k']); kv_b = m(kv['boolq']); kv_r = m(kv['msmarco'])
        vr   = vram(bl['gsm8k']) or '--'

        row = (
            f"\\midrule\n"
            f"\\multirow{{2}}{{*}}{{\\shortstack[l]{{{mname}\\\\({param}B)}}}}\n"
            f"  & BF16 Baseline  & {fmt(bl_g)} & --   & {fmt(bl_b)} & --   & {fmt(bl_r)} & --      & {vr} \\\\\n"
            f"  & KV-Quant 4-bit & {fmt(kv_g)} & {delta(bl_g,kv_g)} & {fmt(kv_b)} & {delta(bl_b,kv_b)} & {fmt(kv_r)} & {delta_rl(bl_r,kv_r)} & {vr} \\\\"
        )
        rows.append((mkey, mname, param, row))
    return rows

def make_table2_rows(results, existing_tex):
    rows = []
    for mkey, (mname, param) in MODELS.items():
        if mname in existing_tex:
            continue
        mdata = results.get(mkey, {})
        bl = mdata.get('baseline', {})
        kv = mdata.get('kv_compress', {})
        if not (bl.get('gsm8k') and bl.get('boolq') and bl.get('msmarco')):
            continue
        tg_b = tput(bl['gsm8k']); tb_b = tput(bl['boolq']); tm_b = tput(bl['msmarco'])
        if not tg_b: continue
        tg_k = tput(kv.get('gsm8k',{})); tb_k = tput(kv.get('boolq',{})); tm_k = tput(kv.get('msmarco',{}))

        row = (
            f"\\midrule\n"
            f"\\multirow{{2}}{{*}}{{\\shortstack[l]{{{mname}\\\\({param}B)}}}}\n"
            f"  & BF16 Baseline  & {fmt(tg_b)} & {fmt(tb_b)} & {fmt(tm_b)} \\\\\n"
            f"  & KV-Quant 4-bit & {fmt(tg_k)} & {fmt(tb_k)} & {fmt(tm_k)} \\\\"
        )
        rows.append((mkey, mname, row))
    return rows

# ─── Main update ─────────────────────────────────────────────────────────────
def update_overleaf(results):
    run("git pull --rebase -q", cwd=OL)
    with open(TEX) as f:
        tex = f.read()

    original = tex
    added = []

    # Table 1
    t1_anchor = "\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\n  \\small\n  \\item[a] KV-Quant:"
    for mkey, mname, param, row in make_table1_rows(results, tex):
        tex = tex.replace(t1_anchor, row + "\n" + t1_anchor, 1)
        added.append(f"{mkey} Table1")
        print(f"  ✓ Added {mname} ({param}B) to Table 1")

    # Table 2 — find its \bottomrule (second one, before \end{table}\n\n\n% ---- Table 3)
    t2_anchor = "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\n% ---- Table 3:"
    for mkey, mname, row in make_table2_rows(results, tex):
        tex = tex.replace(t2_anchor, row + "\n" + t2_anchor, 1)
        added.append(f"{mkey} Table2")
        print(f"  ✓ Added {mname} to Table 2")

    if tex == original:
        print("  No new complete result sets to add.")
        return False

    with open(TEX, 'w') as f:
        f.write(tex)

    msg = "Auto-update results: " + ", ".join(added)
    out = run(f'git add ICPR_2026_LaTeX_Templates/results_tables.tex && git commit -m "{msg}" && git push origin master', cwd=OL)
    last = [l for l in out.splitlines() if l.strip()][-1] if out.strip() else "no output"
    print(f"  → Overleaf push: {last}")
    return True

# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    loop = '--loop' in sys.argv
    interval = 90

    print("=" * 55)
    print("  ICPR Monitor & Overleaf Auto-Update")
    print("=" * 55)

    iteration = 0
    while True:
        iteration += 1
        ts = time.strftime('%H:%M:%S')
        print(f"\n[{ts}] Poll #{iteration}")

        results = fetch()

        # Progress snapshot
        prog = ssh("tmux capture-pane -t icpr_missing -p -S -3 2>/dev/null | grep -E '[0-9]+%.*it|Results written' | grep -v HTTP | tail -2")
        if prog:
            print(f"  Active: {prog.strip()}")

        gpu = ssh("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null")
        if gpu:
            p = [x.strip() for x in gpu.split(',')]
            print(f"  GPU: {p[0]}% util · {int(p[1])//1024}GB · {p[2]}°C")

        # Summarize what's done
        for mkey in ['qwen3_8b','phi4_mini','llama3_8b','gemma3_12b']:
            mdata = results.get(mkey, {})
            for comp in ['baseline','kv_compress']:
                d = mdata.get(comp, {})
                done = [b for b in ['gsm8k','boolq','msmarco'] if b in d]
                if done:
                    vals = [f"{b}={'%.1f%%' % m(d[b]) if d[b].get('accuracy') else d[b].get('rouge_l','?')}" for b in done]
                    print(f"  {mkey}/{comp}: {', '.join(vals)}")

        update_overleaf(results)

        if not loop:
            break

        # Check if all target experiments done
        target_done = all(
            results.get(mk, {}).get(comp, {}).get('gsm8k')
            for mk in ['llama3_8b','gemma3_12b']
            for comp in ['baseline','kv_compress']
        )
        if target_done:
            print("\n✅ All Llama + Gemma experiments complete! Final Overleaf update done.")
            break

        print(f"  Sleeping {interval}s...")
        time.sleep(interval)
