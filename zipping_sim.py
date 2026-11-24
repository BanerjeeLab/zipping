#!/usr/bin/env python3
"""
Figure 6I-K: simulated response of the de novo bicellular junction to an induced
calcium flash (calcium-gated actomyosin-clutch model; Supplementary Information).

Model (equation numbers refer to the Supplementary Information)
---------------------------------------------------------------
  calcium (induced flash)   c(t) = A Theta(t) [ w exp(-t/tau_fast) + (1-w) exp(-t/tau_slow) ]   (S13)
  F-actin                   tau_f df/dt = -(f - f0) + beta c(t)                                   (S8)
  contractility             a(t) = m f(t)            (m = steady Myosin II)                       (S9)
  clutch engagement         alpha(t) = chi s(a),  s(a) = a^n / (a^n + a*^n)                       (S10)
  shoulder tension / force  lambda1 = lambda0 + kappa a,  F = 2 lambda1 cos(theta1) - lambda       (S11)
  elongation                gamma dl/dt = [alpha F]_+   ([x]_+ = max(x,0); inactive here, F > 0)  (S12)

Quantities plotted
------------------
  s_pre, s_post : slopes of least-squares linear fits to l(t) over -2 <= t <= 0 and 0 <= t <= 2 min,
                  the same definition used for the experimental pre/post-flash slopes.
  Delta s       : s_post - s_pre, the flash-induced increase in elongation rate                   (S14)
  flash duration: tau_slow, the decay time of the slow component in (S13); tau_fast and w fixed.

Panels
------
  I  single reference flash (A = 1, tau_slow = 1.5 min): c, f, alpha and l versus time, with the
     pre/post linear fits
  J  s_pre and s_post for 10 flashes with A ~ U(0.6, 1.5), tau_slow ~ U(0.5, 3.0) min, f0 ~ U(0.9, 1.1)
     and 0.03 um measurement noise on l (seeded, reproducible)
  K  Delta s versus tau_slow for A = 0.5, 1.0, 1.5

Units: time in minutes, length in micrometres; other quantities dimensionless. Parameters are
illustrative (Table S1), not fitted.

Usage
-----
  python3 zipping_sim.py
Writes three figures, each as .svg / .pdf / .png, to the current directory:
  Figure6I_reference_flash, Figure6J_ensemble_slopes, Figure6K_duration_dependence.
Requires numpy and matplotlib.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams.update({
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
})
C_CA, C_F, C_ZO, C_L, C_PRE, C_POST = "#D63384", "#2CA02C", "#1F5FBF", "#111111", "#9A9A9A", "#E07A00"

# =============================================================== model
# Parameters
P = dict(
    tau_f=0.45,    # F-actin response time (min)
    f0=1.0,        # baseline F-actin activity
    beta=1.5,      # calcium-to-F-actin gain
    m=1.0,         # steady Myosin II level
    chi=1.0,       # standing clutch competence (engaged remodeling vertex)
    n=3,           # engagement switch steepness
    astar=1.8,     # engagement threshold
    lam0=0.50,     # shoulder tension at zero contractile activity
    kappa=0.60,    # shoulder tension gain with contractile activity
    cos_th=0.75,   # cos(theta1), acute remodeling-vertex angle
    lam_main=0.72, # tension of the de novo junction (lambda)
    gamma=1.0,     # friction coefficient
    V0=1.0,        # velocity scale (um/min per unit force); with gamma = 1 sets 0.1-1 um/min
    l0=5.0,        # initial de novo junction length (um)
)
TAU_FAST, W_FAST = 0.12, 0.5   # fast component of the induced flash (min) and its weight, held fixed


def calcium(t, A=1.0, dur=1.5, tau_fast=TAU_FAST, w=W_FAST):
    """Induced flash at t = 0 (Eq. S13): instantaneous rise, bi-exponential decay.
    'dur' is tau_slow, the decay time of the slow component, used as the flash duration."""
    c = np.zeros_like(t)
    on = t >= 0
    c[on] = A * (w * np.exp(-t[on] / tau_fast) + (1 - w) * np.exp(-t[on] / dur))
    return c


def s(a, n, astar):
    """Saturating engagement switch (Eq. S10)."""
    return a**n / (a**n + astar**n)


def simulate(t, A=1.0, dur=1.5, chi=None, f0=None, noise=0.0, rng=None, **kw):
    """Integrate Eqs. S8-S12 on the time grid t (explicit Euler) for one induced flash.
    Returns a dict with c, f, a, alpha, v (um/min) and l (um)."""
    p = dict(P); p.update(kw)
    if chi is not None: p["chi"] = chi
    if f0 is not None: p["f0"] = f0
    dt = t[1] - t[0]
    c = calcium(t, A, dur)
    f = np.empty_like(t); f[0] = p["f0"]
    for i in range(1, t.size):                                   # Eq. S8
        f[i] = f[i-1] + dt / p["tau_f"] * (-(f[i-1] - p["f0"]) + p["beta"] * c[i-1])
    a = p["m"] * f                                               # Eq. S9
    alpha = p["chi"] * s(a, p["n"], p["astar"])                  # Eq. S10
    lam1 = p["lam0"] + p["kappa"] * a                            # Eq. S11
    F = 2 * lam1 * p["cos_th"] - p["lam_main"]
    v = np.clip(p["V0"] * alpha * F / p["gamma"], 0, None)       # Eq. S12 (rectified; F > 0 here)
    l = p["l0"] + np.cumsum(v) * dt
    if noise > 0 and rng is not None:                            # measurement noise on length (panel J)
        l = l + rng.normal(0, noise, size=l.size)
    return dict(c=c, f=f, a=a, alpha=alpha, v=v, l=l)


def slopes(t, l, win=2.0):
    """(s_pre, s_post): least-squares slopes of l(t) over [-win, 0] and [0, win] (Eq. S14 ingredients)."""
    pre = (t >= -win) & (t <= 0)
    post = (t >= 0) & (t <= win)
    s_pre = np.polyfit(t[pre], l[pre], 1)[0]
    s_post = np.polyfit(t[post], l[post], 1)[0]
    return s_pre, s_post


t = np.arange(-2.0, 2.0 + 1e-9, 0.005)   # min; flash at t = 0


def inline_svg_uses(path):
    """Replace every <use xlink:href="#id"> in a matplotlib SVG by a translated copy of the referenced
    <path>, so the file contains no <defs>/<use> symbols (Adobe Illustrator drops those on import)."""
    src = open(path).read()
    defs = {m.group(1): m.group(0) for m in re.finditer(r'<path id="([^"]+)"[^>]*/>', src)}
    def repl(m):
        attrs = m.group(1)
        href = re.search(r'xlink:href="#([^"]+)"', attrs).group(1)
        x = re.search(r'\bx="([^"]+)"', attrs); y = re.search(r'\by="([^"]+)"', attrs)
        style = re.search(r'style="([^"]*)"', attrs)
        p = re.sub(r' id="[^"]+"', '', defs[href])
        if style:
            p = re.sub(r' style="[^"]*"', '', p).replace('<path', f'<path style="{style.group(1)}"', 1)
        return f'<g transform="translate({x.group(1) if x else 0},{y.group(1) if y else 0})">{p}</g>'
    out = re.sub(r'<use ([^>]*)/>', repl, src)
    out = re.sub(r'<path id="[^"]+"[^>]*/>\s*', '', out)
    open(path, 'w').write(out)


# =============================================================== figures
H_IN = 2.9
MARG = dict(top=0.93, bottom=0.15)


def save(fig, base):
    fig.savefig(base + ".svg"); fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    inline_svg_uses(base + ".svg")


# ---- I: single reference flash, traces
figI = plt.figure(figsize=(2.75, H_IN))
sub = gridspec.GridSpec(4, 1, hspace=0.12, left=0.20, right=0.97, **MARG)
ref = simulate(t, A=1.0, dur=1.5)
sp, so = slopes(t, ref["l"])
axs = [figI.add_subplot(sub[k]) for k in range(4)]
axs[0].plot(t, ref["c"], color=C_CA);     axs[0].set_ylabel("Ca$^{2+}$", color=C_CA, labelpad=2)
axs[1].plot(t, ref["f"], color=C_F);      axs[1].set_ylabel("F-actin", color=C_F, labelpad=2)
axs[2].plot(t, ref["alpha"], color=C_ZO); axs[2].set_ylabel("α (ZO-1)", color=C_ZO, labelpad=2)
axs[3].plot(t, ref["l"], color=C_L);      axs[3].set_ylabel("Length (µm)", color=C_L, labelpad=2)
# pre/post least-squares fits (same windows as slopes())
tp = np.linspace(-2, 0, 2); to = np.linspace(0, 2, 2)
lp = np.polyval(np.polyfit(t[t <= 0], ref["l"][t <= 0], 1), tp)
lo = np.polyval(np.polyfit(t[t >= 0], ref["l"][t >= 0], 1), to)
axs[3].plot(tp, lp, "--", color=C_PRE, lw=0.9); axs[3].plot(to, lo, "--", color=C_POST, lw=0.9)
axs[3].text(-1.0, ref["l"][0] + 0.45*(ref["l"][-1] - ref["l"][0]), f"$s_{{\\rm pre}}$ = {sp:.2f}", fontsize=5.5, color=C_PRE, ha="center")
axs[3].text(1.0, ref["l"][0] + 0.15*(ref["l"][-1] - ref["l"][0]), f"$s_{{\\rm post}}$ = {so:.2f} µm/min", fontsize=5.5, color=C_POST, ha="center")
for a_ in axs:
    a_.axvline(0, color="#999", lw=0.6, ls=":"); a_.set_xlim(-2, 2)
    a_.tick_params(labelbottom=False, length=2)
for a_ in axs[:3]: a_.set_yticks([])
axs[3].tick_params(labelbottom=True); axs[3].set_xlabel("Time from stimulation (min)")
axs[0].text(0.52, 1.04, "stimulation", transform=axs[0].transAxes, fontsize=5.5, color=C_CA, ha="left")

save(figI, "Figure6I_reference_flash")

# ---- J: s_pre and s_post for an ensemble of induced flashes
figJ = plt.figure(figsize=(1.85, H_IN))
figJ.subplots_adjust(left=0.30, right=0.96, **MARG)
axJ = figJ.add_subplot(111)
rng = np.random.default_rng(3)            # fixed seed: the ensemble is reproducible
pairs = []
for k in range(10):
    A = rng.uniform(0.6, 1.5); dur = rng.uniform(0.5, 3.0); f0 = rng.uniform(0.9, 1.1)
    r_ = simulate(t, A=A, dur=dur, f0=f0, noise=0.03, rng=rng)
    pairs.append(slopes(t, r_["l"]))
pairs = np.array(pairs)
for pr, po in pairs: axJ.plot([0, 1], [pr, po], "-o", color="#777", ms=2.5, lw=0.7, mfc="white", alpha=0.9)
axJ.plot([0, 1], pairs.mean(0), "-o", color=C_POST, ms=3.5, lw=1.6, zorder=5)
axJ.set_xticks([0, 1]); axJ.set_xticklabels(["pre", "post"]); axJ.set_xlim(-0.35, 1.35)
axJ.set_ylabel("Elongation slope $s$ (µm/min)"); axJ.axhline(0, color="#bbb", lw=0.5)
axJ.set_xlabel("induced flash")
axJ.set_ylim(-0.02, 0.72)
axJ.text(0.5, 0.99, "n = 10 simulated flashes\n(varied amplitude & duration)", transform=axJ.transAxes,
         ha="center", va="top", fontsize=5.5, color="#555")

save(figJ, "Figure6J_ensemble_slopes")

# ---- K: Delta s versus flash duration for three amplitudes
figK = plt.figure(figsize=(2.25, H_IN))
figK.subplots_adjust(left=0.24, right=0.96, **MARG)
axK = figK.add_subplot(111)
durs = np.linspace(0.15, 4.0, 40)
for A, col, lab in [(0.5, "#F1A7C8", "A = 0.5"), (1.0, C_CA, "A = 1.0"), (1.5, "#7A0B3E", "A = 1.5")]:
    ds = []
    for d in durs:
        s_pre, s_post = slopes(t, simulate(t, A=A, dur=d)["l"])
        ds.append(s_post - s_pre)                                # Eq. S14
    axK.plot(durs, ds, color=col, label=lab)
axK.set_xlabel("Flash duration $\\tau_{\\rm slow}$ (min)")
axK.set_ylabel("$\\Delta s = s_{\\rm post} - s_{\\rm pre}$ (µm/min)")
axK.legend(frameon=False, loc="upper left", handlelength=1.4, title="flash amplitude", title_fontsize=5.5, borderaxespad=0.2)
axK.set_xlim(0, 4); axK.axhline(0, color="#bbb", lw=0.5)

save(figK, "Figure6K_duration_dependence")

print(f"reference flash: s_pre = {sp:.3f}, s_post = {so:.3f} um/min | "
      f"ensemble mean s_pre = {pairs[:,0].mean():.2f}, s_post = {pairs[:,1].mean():.2f} | "
      f"F at baseline = {2*(P['lam0']+P['kappa']*P['f0'])*P['cos_th']-P['lam_main']:.2f} (> 0, rectifier inactive)")
