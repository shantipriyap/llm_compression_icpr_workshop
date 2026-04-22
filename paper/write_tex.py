#!/usr/bin/env python3
"""One-shot rewriter for results_tables.tex with real benchmark numbers."""
TEX = r"""% ============================================================
% ICPR 2026 Workshop — LLM Compression Benchmark Results
% Real numbers from 200-sample evaluation on NVIDIA RTX PRO 6000 Blackwell
% ============================================================

% Required packages in preamble:
%   \usepackage{booktabs}
%   \usepackage{multirow}
%   \usepackage{threeparttable}
%   \usepackage{xcolor}


% ── Table 1: Main English Benchmark Results ─────────────────────
\begin{table*}[t]
\centering
\caption{English benchmark results for Qwen3-8B and Phi-4-Mini under
different compression strategies (200-sample subsets).
$\Delta$ = absolute change vs.\ BF16 baseline.
``\textemdash'' = evaluation pending.}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccc}
\toprule
\multirow{2}{*}{\textbf{Model}} &
\multirow{2}{*}{\textbf{Method}} &
\multicolumn{2}{c}{\textbf{GSM8K}} &
\multicolumn{2}{c}{\textbf{BoolQ}} &
\multicolumn{2}{c}{\textbf{MS MARCO}} &
\textbf{VRAM} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
& & Acc.\,(\%) & $\Delta$ & Acc.\,(\%) & $\Delta$ & ROUGE-L & $\Delta$ & (GB) \\
\midrule
\multirow{4}{*}{Qwen3-8B}
  & BF16 Baseline          & \textbf{19.0} & \textemdash & \textbf{87.5} & \textemdash & 0.0616 & \textemdash & 15.3 \\
  & KV-Quant 4-bit         & 19.0 & 0.0 & 87.5 & 0.0 & 0.0620 & +0.6\% & 15.3 \\
  & AWQ 4-bit              & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash \\
  & GPTQ 4-bit             & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash \\
\midrule
\multirow{4}{*}{\shortstack[l]{Phi-4-Mini\\(3.8B)}}
  & BF16 Baseline          & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash \\
  & KV-Quant 4-bit         & \textbf{85.5} & \textemdash & \textbf{83.5} & \textemdash & 0.1398 & \textemdash & 7.2 \\
  & AWQ 4-bit              & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash \\
  & GPTQ 4-bit             & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash & \textemdash \\
\bottomrule
\end{tabular}
\begin{tablenotes}
  \small
  \item Phi-4-Mini baseline and AWQ/GPTQ evaluations in progress for both models.
  \item KV-Quant: 4-bit asymmetric per-group quantization (group=64), applied via forward hooks.
\end{tablenotes}
\end{table*}


% ── Table 2: Inference Throughput ───────────────────────────────
\begin{table}[t]
\centering
\caption{Inference throughput (tokens/sec) on NVIDIA RTX PRO 6000 Blackwell (102\,GB VRAM).}
\label{tab:throughput}
\small
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Method} & \textbf{GSM8K} & \textbf{BoolQ} & \textbf{MS MARCO} \\
               &                 & (tok/s)        & (tok/s)        & (tok/s) \\
\midrule
\multirow{2}{*}{Qwen3-8B}
  & BF16 Baseline  & 59.4 & 54.9 & 58.9 \\
  & KV-Quant 4-bit & 59.4 & 53.7 & 57.5 \\
\midrule
Phi-4-Mini (3.8B) & KV-Quant 4-bit & 96.9 & 75.6 & 93.9 \\
\bottomrule
\end{tabular}
\end{table}


% ── Table 3: Multilingual Robustness Results ─────────────────────
\begin{table*}[t]
\centering
\caption{Multilingual robustness evaluation on Hindi and Odia.
IndicSentiment: binary sentiment accuracy (\%).
Random-chance baseline for binary classification = 50\%.
``\textemdash'' = evaluation pending.}
\label{tab:multilingual_results}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llcccccc}
\toprule
\multirow{2}{*}{\textbf{Model}} &
\multirow{2}{*}{\textbf{Method}} &
\multirow{2}{*}{\textbf{MGSM-Hi}} &
\multicolumn{2}{c}{\textbf{IndicQA (F1)}} &
\multicolumn{2}{c}{\textbf{IndicSentiment (\%)}} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
& & & Hindi & Odia & Hindi & Odia \\
\midrule
\multirow{2}{*}{Qwen3-8B}
  & BF16 Baseline          & \textemdash & \textemdash & \textemdash & 0.0$^{*}$ & 0.0$^{*}$ \\
  & KV-Quant 4-bit         & \textemdash & \textemdash & \textemdash & 0.0$^{*}$ & 0.0$^{*}$ \\
\midrule
\multirow{2}{*}{\shortstack[l]{Phi-4-Mini\\(3.8B)}}
  & BF16 Baseline          & \textemdash & \textemdash & \textemdash & 50.0$^{\dagger}$ & 50.0$^{\dagger}$ \\
  & KV-Quant 4-bit         & \textemdash & \textemdash & \textemdash & 50.0$^{\dagger}$ & 50.0$^{\dagger}$ \\
\bottomrule
\end{tabular}
\begin{tablenotes}
  \small
  \item[$*$] Qwen3-8B generates extended reasoning tokens and does not produce a
        parseable ``Positive''/``Negative'' response for Indic-script input; 0\% under exact-match.
  \item[$\dagger$] Phi-4-Mini does not follow English output instructions for Indic-script input;
        50\% = random-chance baseline for binary classification.
  \item These results highlight instruction-following as a key axis of multilingual robustness,
        independent of compression method.
  \item MGSM-Hi and IndicQA results pending dataset compatibility resolution.
\end{tablenotes}
\end{table*}


% ── Table 4: Evaluation Benchmark Summary ────────────────────────
\begin{table}[t]
\centering
\caption{Benchmarks used in this study (200 samples each).}
\label{tab:benchmarks}
\small
\begin{tabular}{lllll}
\toprule
\textbf{Benchmark} & \textbf{Lang.} & \textbf{Task} & \textbf{Metric} & $N$ \\
\midrule
GSM8K~\cite{cobbe2021gsm8k}                  & EN     & Math reasoning & Acc.\,(EM)  & 200 \\
BoolQ~\cite{clark2019boolq}                  & EN     & Boolean QA     & Accuracy    & 200 \\
MS MARCO~\cite{nguyen2016ms}                 & EN     & Passage QA     & ROUGE-L     & 200 \\
MGSM~\cite{shi2023language}                  & Hi     & Math reasoning & Acc.\,(EM)  & 200 \\
IndicQA~\cite{doddapaneni2023towards}        & Hi, Or & Reading comp.  & F1          & 200 \\
IndicSentiment~\cite{doddapaneni2023towards} & Hi, Or & Sentiment      & Accuracy    & 200 \\
\bottomrule
\end{tabular}
\end{table}


% ── BibTeX entries (add to your .bib file) ───────────────────────
% @article{shi2023language,
%   title   = {Language Models are Multilingual Chain-of-Thought Reasoners},
%   author  = {Shi, Freda and others},
%   journal = {ICLR},
%   year    = {2023}
% }
% @inproceedings{doddapaneni2023towards,
%   title   = {Towards Leaving No Indic Language Behind},
%   author  = {Doddapaneni, Sumanth and others},
%   booktitle = {ACL},
%   year    = {2023}
% }
% @article{cobbe2021gsm8k,
%   title   = {Training Verifiers to Solve Math Word Problems},
%   author  = {Cobbe, Karl and others},
%   year    = {2021},
%   journal = {arXiv:2110.14168}
% }
% @inproceedings{clark2019boolq,
%   title   = {{BoolQ}: Exploring the Surprising Difficulty of Natural Yes/No Questions},
%   author  = {Clark, Christopher and others},
%   booktitle = {NAACL},
%   year    = {2019}
% }
% @article{nguyen2016ms,
%   title   = {{MS MARCO}: A Human Generated Machine Reading Comprehension Dataset},
%   author  = {Nguyen, Tri and others},
%   year    = {2016},
%   journal = {arXiv:1611.09268}
% }
"""

out = "/Users/shantipriya/work/icpr/paper/results_tables.tex"
with open(out, "w") as f:
    f.write(TEX)

lines = TEX.count("\n")
tables = TEX.count(r"\begin{table")
print(f"Written {lines} lines, {tables} tables to {out}")
