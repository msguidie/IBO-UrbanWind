"""The run's results-table row as TeX: the six relative-L2 channels and
the five cost columns.
"""
from __future__ import annotations

from pathlib import Path

TEMPLATE = r"""\begin{table}[htbp]
\centering
\caption{UrbanWind test performance and cost}
\label{tab:urbanwind_{LABEL}}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc|ccccc}
\toprule
\multirow{2}{*}{\textsc{Model}} & \multicolumn{6}{c|}{\textsc{Relative $L_2$} $\downarrow$} & \multirow{2}{*}{\shortstack{Params\\(M)}} & \multirow{2}{*}{\shortstack{Peak Mem\\(GiB)}} & \multirow{2}{*}{\shortstack{Avg Latency\\(s/case)}} & \multirow{2}{*}{\shortstack{Train FLOPs\\(TF/step)}} & \multirow{2}{*}{\shortstack{Test FLOPs\\(TF/case)}} \\
\cmidrule(lr){2-7}
 & $U_x$ & $U_y$ & $U_z$ & $p$ & $\nu_t$ & $p_{surf}$ & & & & & \\
\midrule
{ROW}
\bottomrule
\end{tabular}%
}
\end{table}
"""


def _fmt(v, digits=4):
    if v is None:
        return "/"
    return f"{float(v):.{digits}f}"


def render_tex(run_dir: Path, row_name: str, merged: dict,
               tele: dict) -> Path:
    chan = merged["rel_l2"]
    std = merged.get("rel_l2_std") or {}
    cells = [row_name]
    for ch in ("Ux", "Uy", "Uz", "p", "nut", "p_surf"):
        # stochastic producers (AB-UPT's anchor draw): mean +- std over the
        # independent inference draws, std in small print
        cells.append(_fmt(chan.get(ch)) + (
            r"{\scriptsize$\pm$" + _fmt(std[ch]) + "}" if ch in std else ""))
    cells.append(_fmt(tele.get("params_trainable", 0) / 1e6, 2))
    peak = tele.get("peak_train_mem_bytes")
    cells.append(_fmt(peak / 2**30, 2) if peak else "/")
    lat = (tele.get("latency") or {}).get("avg_s_per_case")
    cells.append(_fmt(lat, 3) if lat is not None else "/")
    tf = tele.get("train_flops_per_opt_step")
    cells.append(_fmt(tf / 1e12, 2) if tf else "/")
    xf = tele.get("test_flops_avg_fullcase")
    cells.append(_fmt(xf / 1e12, 2) if xf else "/")
    row = " & ".join(cells) + r" \\"
    label = row_name.lower().replace(" ", "_").replace("-", "_")
    tex = TEMPLATE.replace("{ROW}", row).replace("{LABEL}", label)
    out = Path(run_dir) / "table.tex"
    tmp = str(out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(tex)
    Path(tmp).replace(out)
    return out
