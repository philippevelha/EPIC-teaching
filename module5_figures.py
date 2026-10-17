"""
Figures for Module 5 - light-matter interaction & optical amplification.

Run:  python module5_figures.py
Writes m5_*.png next to this script (and m5_gain_data.json for the
interactive gain slide).

Physics models (deliberately simple, for teaching):
  * Band structures: local empirical pseudopotential method
    (Cohen & Bergstresser, Phys. Rev. 141, 789 (1966)), no spin-orbit.
  * Gain spectrum: bulk parabolic bands, k-conserving transitions,
    quasi-Fermi levels from Fermi-Dirac statistics, Gaussian broadening.
  * SOA: homogeneously broadened gain saturation g = g0/(1+P/Psat).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
C_CB, C_VB, C_PH, C_ACC = "#1f6fb4", "#c0392b", "#e67e22", "#2e8b57"


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# 1. Empirical pseudopotential band structures (Si, Ge, GaAs)
# ---------------------------------------------------------------------------
RY = 13.605693  # eV
BOHR = 0.529177  # Angstrom
EPM = {  # a [A], (VS3, VS8, VS11), (VA3, VA4, VA11)   [Ry]
    "Si":   (5.43, (-0.21, 0.04, 0.08), (0.0, 0.0, 0.0)),
    "Ge":   (5.66, (-0.23, 0.01, 0.06), (0.0, 0.0, 0.0)),
    "GaAs": (5.64, (-0.23, 0.01, 0.06), (0.07, 0.05, 0.01)),
}


def g_vectors(gmax2=21):
    n = range(-4, 5)
    gs = []
    for i in n:
        for j in n:
            for k in n:
                # fcc reciprocal lattice = bcc: all even or all odd
                if (i % 2 == j % 2 == k % 2) and i * i + j * j + k * k <= gmax2:
                    gs.append((i, j, k))
    return np.array(gs, float)


def epm_bands(material, kpts, nb=8):
    a_ang, VS, VA = EPM[material]
    a = a_ang / BOHR
    G = g_vectors()
    dG = G[:, None, :] - G[None, :, :]
    d2 = np.rint((dG ** 2).sum(-1)).astype(int)
    phase = 2 * np.pi * (dG @ np.array([1, 1, 1]) / 8.0)
    vs = np.zeros_like(phase); va = np.zeros_like(phase)
    for g2, s, an in zip((3, 8, 11), VS, (0, 0, 0)):
        vs[d2 == g2] = s
    for g2, an in zip((3, 4, 11), VA):
        va[d2 == g2] = an
    V = vs * np.cos(phase) + 1j * va * np.sin(phase)
    out = []
    for k in kpts:
        T = ((k[None, :] + G) ** 2).sum(-1) * (2 * np.pi / a) ** 2
        H = V + np.diag(T)
        out.append(np.linalg.eigvalsh(H)[:nb])
    return np.array(out) * RY


def kpath(npts=80):
    P = {"L": (0.5, 0.5, 0.5), "Γ": (0, 0, 0), "X": (1, 0, 0),
         "K": (0.75, 0.75, 0)}
    seq = ["L", "Γ", "X", "K", "Γ"]
    ks, ticks = [], [0.0]
    for p, q in zip(seq[:-1], seq[1:]):
        A, B = np.array(P[p]), np.array(P[q])
        ks.extend(A + t * (B - A) for t in np.linspace(0, 1, npts, endpoint=False))
        ticks.append(ticks[-1] + np.linalg.norm(B - A))
    ks.append(np.array(P[seq[-1]]))
    ks = np.array(ks)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))]
    return ks, d, ticks, seq


def fig_band_structures():
    ks, d, ticks, labels = kpath()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, mat in zip(axs, ["Si", "Ge", "GaAs"]):
        E = epm_bands(mat, ks)
        vbm = E[:, 3].max()
        E -= vbm
        cb = E[:, 4]
        icb = np.argmin(cb)
        for b in range(8):
            ax.plot(d, E[:, b], color=C_VB if b < 4 else C_CB, lw=1.8)
        ax.axhspan(0, cb.min(), color="0.92")
        ivb = np.argmax(E[:, 3])
        ax.plot(d[ivb], 0, "o", color=C_VB, ms=9)
        ax.plot(d[icb], cb[icb], "o", color=C_CB, ms=9)
        direct = abs(d[icb] - d[ivb]) < 1e-6
        if direct:
            ax.annotate("", (d[ivb], cb[icb]), (d[ivb], 0),
                        arrowprops=dict(arrowstyle="<->", color=C_PH, lw=2.5))
        else:
            ax.annotate("", (d[icb], cb[icb]), (d[ivb], 0),
                        arrowprops=dict(arrowstyle="->", color=C_ACC, lw=2.5,
                                        connectionstyle="arc3,rad=-0.2"))
        kind = "direct" if direct else "indirect"
        ax.set_title(f"{mat}:  E$_g$ ≈ {cb.min():.2f} eV ({kind}, model)")
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
        for t in ticks:
            ax.axvline(t, color="0.75", lw=0.8)
        ax.set_xlim(d[0], d[-1]); ax.set_ylim(-5, 5.5)
    axs[0].set_ylabel("Energy  E − E$_v$ (eV)")
    axs[0].text(ticks[0] + 0.05, -4.6, "[111]", color="0.35")
    axs[0].text(ticks[1] + 0.05, -4.6, "[100]", color="0.35")
    fig.text(0.5, -0.02, "Empirical pseudopotential (Cohen–Bergstresser), no spin–orbit: "
             "gap values are indicative, the valley positions are what matters",
             ha="center", fontsize=10, color="0.4")
    save(fig, "m5_band_structures.png")


def fig_direction_cut():
    """Si & GaAs conduction band along [100] and [111] from the same Γ point."""
    n = 120
    t = np.linspace(0, 1, n)
    k100 = np.outer(t, [1, 0, 0])
    k111 = np.outer(t, [0.5, 0.5, 0.5])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, mat in zip(axs, ["Si", "GaAs"]):
        E100 = epm_bands(mat, k100); E111 = epm_bands(mat, k111)
        vbm = max(E100[:, 3].max(), E111[:, 3].max())
        x100 = t * 1.0; x111 = -t * np.linalg.norm([0.5] * 3)
        for E, x, lab, ls in ((E100, x100, "[100] → X", "-"), (E111, x111, "[111] → L", "-")):
            ax.plot(x, E[:, 4] - vbm, color=C_CB, lw=2.2, ls=ls)
            ax.plot(x, E[:, 3] - vbm, color=C_VB, lw=2.2, ls=ls)
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set_xticks([x111[-1], 0, 1]); ax.set_xticklabels(["L\n[111]", "Γ", "X\n[100]"])
        cbmin = min(E100[:, 4].min(), E111[:, 4].min()) - vbm
        if mat == "Si":
            i = np.argmin(E100[:, 4])
            ax.annotate(f"CB minimum at k ≈ 0.85·(2π/a)\nalong ⟨100⟩ (6 valleys)",
                        (x100[i], E100[i, 4] - vbm), (0.15, 3.3),
                        arrowprops=dict(arrowstyle="->"), fontsize=11)
        else:
            ax.annotate("CB minimum at Γ (k = 0)\nsame k as VB maximum",
                        (0, cbmin), (-0.85, 3.6), arrowprops=dict(arrowstyle="->"), fontsize=11)
        ax.set_title(mat); ax.set_ylim(-2.5, 4.5)
        ax.set_xlabel("wave-vector  k  (units of 2π/a)")
    axs[0].set_ylabel("E − E$_v$ (eV)")
    save(fig, "m5_band_directions.png")


def fig_valleys():
    """Constant-energy surfaces of the CB minimum in the (k_x, k_z) and (110) planes."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    # Si: 6 ellipsoids along <100>, ml = 0.92, mt = 0.19 -> semi-axes ∝ sqrt(m)
    ax = axs[0]
    k0, sl, st = 0.85, 0.16 * np.sqrt(0.92 / 0.19) / 2.2, 0.16 / 2.2
    for (x, y, w, h) in [(k0, 0, sl, st), (-k0, 0, sl, st), (0, k0, st, sl), (0, -k0, st, sl)]:
        ax.add_patch(Ellipse((x, y), 2 * w, 2 * h, color=C_CB, alpha=0.6))
    ax.text(-1.1, 1.0, "(+2 valleys along ±k$_z$)", fontsize=10, color="0.35")
    ax.set_title("Si: 6 valleys along ⟨100⟩\n(m$_l$=0.92, m$_t$=0.19 m$_0$)")
    ax.set_xlabel("k$_x$  [100]"); ax.set_ylabel("k$_y$  [010]")
    # Ge: 8 half-valleys at L along <111>; in (1-10) plane show [111] directions
    ax = axs[1]
    for ang in (np.deg2rad(54.7), np.deg2rad(180 - 54.7), np.deg2rad(180 + 54.7), np.deg2rad(-54.7)):
        L = 0.866
        x, y = L * np.sin(ang), L * np.cos(ang)
        ax.add_patch(Ellipse((x, y), 2 * 0.06, 2 * 0.28, angle=-np.rad2deg(ang),
                             color=C_CB, alpha=0.6))
    ax.set_title("Ge: valleys at L along ⟨111⟩\n(m$_l$=1.6, m$_t$=0.08 m$_0$)")
    ax.set_xlabel("k along [110]"); ax.set_ylabel("k along [001]")
    # GaAs: one isotropic valley at Γ
    ax = axs[2]
    ax.add_patch(Ellipse((0, 0), 2 * 0.12, 2 * 0.12, color=C_CB, alpha=0.6))
    ax.set_title("GaAs: single isotropic valley at Γ\n(m$_e$=0.067 m$_0$)")
    ax.set_xlabel("k$_x$"); ax.set_ylabel("k$_y$")
    for ax in axs:
        ax.plot(0, 0, "o", color=C_VB, ms=8)
        ax.annotate("VB max (Γ)", (0, 0), (0.05, -0.25), color=C_VB, fontsize=10)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
        ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    fig.text(0.5, -0.03, "k in units of 2π/a.  The photon wave-vector at 1550 nm "
             "(≈ 2π·3.5/λ) is ~10⁻³ of these distances: optical transitions are vertical.",
             ha="center", fontsize=11, color="0.3")
    save(fig, "m5_valleys.png")


# ---------------------------------------------------------------------------
# 2. Direct vs indirect transitions on parabolic bands
# ---------------------------------------------------------------------------
def wavy(ax, x0, y0, x1, y1, color=C_PH, n=6, amp=0.05, lw=2.2, head=True):
    t = np.linspace(0, 1, 300)
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy); ux, uy = dx / L, dy / L
    s = amp * np.sin(2 * np.pi * n * t)
    ax.plot(x0 + dx * t * 0.92 - uy * s, y0 + dy * t * 0.92 + ux * s, color=color, lw=lw)
    if head:
        ax.add_patch(FancyArrowPatch((x0 + dx * 0.88, y0 + dy * 0.88), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=18, color=color))


def fig_direct_indirect():
    k = np.linspace(-1, 1, 300)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    # direct
    ax = axs[0]
    ax.plot(k, 1.0 + 1.6 * k ** 2, color=C_CB, lw=2.5)
    ax.plot(k, -0.9 * k ** 2, color=C_VB, lw=2.5)
    ax.annotate("", (0, 1.0), (0, 0), arrowprops=dict(arrowstyle="<|-", color=C_PH, lw=2.5))
    wavy(ax, 0.15, 0.5, 0.75, 0.5)
    ax.text(0.4, 0.62, "hν ≈ E$_g$", color=C_PH)
    ax.set_title("Direct gap (GaAs, InP, InGaAsP)\nk$_{photon}$ ≈ 0 → vertical, 1st-order process")
    # indirect
    ax = axs[1]
    k0 = 0.75
    ax.plot(k, 1.0 + 2.0 * (k - k0) ** 2, color=C_CB, lw=2.5)
    ax.plot(k, 1.9 + 1.6 * k ** 2, color=C_CB, lw=1.2, ls="--")
    ax.plot(k, -0.9 * k ** 2, color=C_VB, lw=2.5)
    ax.annotate("", (k0, 1.0), (0, 0), arrowprops=dict(arrowstyle="<|-", color="0.4", lw=1.5, ls="--"))
    ax.annotate("", (k0, 1.0), (0.02, 1.02), arrowprops=dict(arrowstyle="<|-", color=C_ACC, lw=2.5))
    ax.annotate("", (0.0, 1.0), (0, 0), arrowprops=dict(arrowstyle="<|-", color=C_PH, lw=2.5))
    ax.text(0.05, 0.84, "phonon (ħq, ħΩ)", color=C_ACC, fontsize=11)
    ax.text(0.45, 0.25, "✗ photon alone:\nΔk ≠ 0", color="0.4", fontsize=11)
    ax.text(-0.95, 2.0, "Γ valley (higher)", color=C_CB, fontsize=11)
    ax.text(-0.95, 0.5, "photon + phonon:\n2nd-order process\n→ ~10³–10⁴× weaker", fontsize=11)
    ax.set_title("Indirect gap (Si, Ge)\nCB minimum at k ≠ 0 → needs momentum ħq")
    for ax in axs:
        ax.set_xlabel("wave-vector  k"); ax.set_ylabel("Energy")
        ax.set_xticks([0]); ax.set_xticklabels(["Γ (k=0)"]); ax.set_yticks([])
        ax.set_ylim(-1.0, 3.2)
    save(fig, "m5_direct_indirect.png")


# ---------------------------------------------------------------------------
# 3. Absorption / spontaneous / stimulated emission
# ---------------------------------------------------------------------------
def fig_three_processes():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    titles = ["Absorption\nrate = B$_{12}$ ρ(ν) N$_1$",
              "Spontaneous emission\nrate = A$_{21}$ N$_2$",
              "Stimulated emission\nrate = B$_{21}$ ρ(ν) N$_2$"]
    for ax, ttl in zip(axs, titles):
        ax.plot([0.3, 0.7], [1, 1], "k", lw=3); ax.plot([0.3, 0.7], [0, 0], "k", lw=3)
        ax.text(0.73, 1, "E$_2$  (CB)", va="center"); ax.text(0.73, 0, "E$_1$  (VB)", va="center")
        ax.set_title(ttl); ax.set_xlim(-0.35, 1.15); ax.set_ylim(-0.4, 1.4); ax.axis("off")
    # absorption
    ax = axs[0]
    wavy(ax, -0.3, 0.5, 0.25, 0.5)
    ax.plot(0.5, 0, "o", mfc="none", mec=C_CB, ms=12, mew=2)
    ax.plot(0.5, 1, "o", color=C_CB, ms=12)
    ax.annotate("", (0.5, 0.9), (0.5, 0.1), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.text(-0.3, 0.62, "hν", color=C_PH)
    # spontaneous
    ax = axs[1]
    ax.plot(0.5, 1, "o", mfc="none", mec=C_CB, ms=12, mew=2)
    ax.plot(0.5, 0, "o", color=C_CB, ms=12)
    ax.annotate("", (0.5, 0.1), (0.5, 0.9), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    wavy(ax, 0.6, 0.45, 1.05, -0.3, n=5)
    ax.text(0.62, 0.62, "random direction,\nphase, polarisation", fontsize=10, color=C_PH)
    # stimulated
    ax = axs[2]
    wavy(ax, -0.3, 0.5, 0.25, 0.5)
    ax.plot(0.5, 1, "o", mfc="none", mec=C_CB, ms=12, mew=2)
    ax.plot(0.5, 0, "o", color=C_CB, ms=12)
    ax.annotate("", (0.5, 0.1), (0.5, 0.9), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    wavy(ax, 0.6, 0.62, 1.1, 0.62, n=5)
    wavy(ax, 0.6, 0.38, 1.1, 0.38, n=5)
    ax.text(0.68, 0.18, "2 identical photons\n(same k, phase, pol.)", fontsize=10, color=C_PH)
    save(fig, "m5_three_processes.png")


def fig_einstein_ratio():
    h, kB, c = 6.62607e-34, 1.380649e-23, 2.99792458e8
    lam = np.logspace(-7, -2, 400)  # 100 nm - 1 cm
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axs[0]
    for T, col in ((300, C_CB), (1000, C_ACC), (5800, C_PH)):
        x = h * c / (lam * kB * T)
        r = 1 / np.expm1(np.clip(x, 1e-9, 700))
        ax.loglog(lam * 1e6, r, lw=2.2, color=col, label=f"T = {T} K")
    ax.axvline(1.55, color="0.5", ls="--"); ax.text(1.75, 1e-38, "1.55 µm", color="0.4")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylim(1e-40, 1e4); ax.set_xlim(0.1, 1e4)
    ax.set_xlabel("wavelength (µm)"); ax.set_ylabel("R$_{stim}$ / R$_{spon}$  (thermal light)")
    ax.set_title("Thermal equilibrium: stimulated ≪ spontaneous\nfor optical frequencies")
    ax.legend(frameon=False, loc="lower right")
    ax.annotate("300 K, 1.55 µm:\n≈ 4×10⁻¹⁴", (1.55, 4e-14), (15, 1e-20),
                arrowprops=dict(arrowstyle="->"), fontsize=11)
    # population ratio
    ax = axs[1]
    E = np.linspace(0, 1.0, 300)
    for T, col in ((300, C_CB), (1000, C_ACC), (5800, C_PH)):
        ax.semilogy(E, np.exp(-E / (8.617e-5 * T)), lw=2.2, color=col, label=f"T = {T} K")
    ax.axvline(0.8, color="0.5", ls="--"); ax.text(0.81, 1e-3, "hν = 0.8 eV\n(1.55 µm)", color="0.4")
    ax.set_ylim(1e-16, 2); ax.set_xlabel("E$_2$ − E$_1$ (eV)"); ax.set_ylabel("N$_2$ / N$_1$ (Boltzmann)")
    ax.set_title("Heating never gives N$_2$ > N$_1$:\ninversion needs pumping")
    ax.legend(frameon=False)
    save(fig, "m5_einstein_ratio.png")


# ---------------------------------------------------------------------------
# 4. Semiconductor gain: quasi-Fermi levels & gain spectrum (bulk InGaAs)
# ---------------------------------------------------------------------------
KT = 0.02585  # eV at 300 K
MAT = dict(Eg=0.75, me=0.041, mh=0.46, alpha0=3.0e4)  # alpha0 [cm^-1 eV^-1/2]


def eff_dos(m):  # cm^-3, 300 K
    return 2.5094e19 * m ** 1.5


def fermi_half(eta):
    f = lambda x: np.sqrt(x) / (1 + np.exp(np.clip(x - eta, -700, 700)))
    return 2 / np.sqrt(np.pi) * quad(f, 0, max(eta, 0) + 40)[0]


def quasi_fermi(N, m):
    """Return (E_F - E_band) in eV for carrier density N [cm^-3]."""
    Nc = eff_dos(m)
    eta = brentq(lambda e: fermi_half(e) - N / Nc, -30, 80)
    return eta * KT


def gain_spectrum(N, E, broad=0.010):
    Eg, me, mh, a0 = MAT["Eg"], MAT["me"], MAT["mh"], MAT["alpha0"]
    mr = me * mh / (me + mh)
    Fc = Eg + quasi_fermi(N, me)          # measured from Ev = 0
    Fv = -quasi_fermi(N, mh)
    x = np.clip(E - Eg, 0, None)
    E2 = Eg + x * mr / me
    E1 = -x * mr / mh
    fc = 1 / (1 + np.exp((E2 - Fc) / KT))
    fv = 1 / (1 + np.exp((E1 - Fv) / KT))
    g = a0 * np.sqrt(x) * (fc - fv)
    if broad:
        dE = E[1] - E[0]
        kx = np.arange(-4 * broad, 4 * broad + dE, dE)
        ker = np.exp(-kx ** 2 / (2 * broad ** 2)); ker /= ker.sum()
        g = np.convolve(g, ker, mode="same")
    return g, Fc - Fv


def fig_fermi_filling():
    """Band filling picture: occupied states under quasi-Fermi levels."""
    # left: schematic (not to scale), both carriers degenerate
    k = np.linspace(-1, 1, 400)
    Eg_s, ac, av, Fc_s, Fv_s = 1.0, 1.4, 0.8, 1.35, -0.25
    Ec = Eg_s + ac * k ** 2
    Ev = -av * k ** 2
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    ax.plot(k, Ec, color=C_CB, lw=2.5); ax.plot(k, Ev, color=C_VB, lw=2.5)
    ax.fill_between(k, Ec, Fc_s, where=Ec < Fc_s, color=C_CB, alpha=0.35)
    ax.fill_between(k, Ev, Fv_s, where=Ev > Fv_s, color="white", ec=C_VB, hatch="//")
    ax.axhline(Fc_s, color=C_CB, ls="--"); ax.axhline(Fv_s, color=C_VB, ls="--")
    ax.text(1.03, Fc_s, "E$_{Fc}$", color=C_CB, va="center")
    ax.text(1.03, Fv_s, "E$_{Fv}$", color=C_VB, va="center")
    kk = 0.22
    ax.annotate("", (kk, Eg_s + ac * kk ** 2), (kk, -av * kk ** 2),
                arrowprops=dict(arrowstyle="<|-", color=C_PH, lw=2.5))
    ax.text(kk + 0.04, 0.45, "full → empty:\nstimulated\nemission wins", color=C_PH, fontsize=11)
    kk = -0.7
    ax.annotate("", (kk, Eg_s + ac * kk ** 2), (kk, -av * kk ** 2),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=2))
    ax.text(-0.98, 0.45, "hν > E$_{Fc}$−E$_{Fv}$:\nstill absorbed", color="0.3", fontsize=11)
    ax.text(0.45, -0.12, "holes", color=C_VB, fontsize=11)
    ax.text(0.2, 1.2, "electrons", color=C_CB, fontsize=11)
    ax.set_ylim(-0.9, 2.3); ax.set_xticks([0]); ax.set_xticklabels(["k = 0"])
    ax.set_yticks([]); ax.set_ylabel("Energy"); ax.set_xlabel("k")
    ax.set_title("Forward-biased junction (schematic)\nvertical transitions, k conserved")
    # right: occupation functions for InGaAs at N = 2e18
    N = 2e18
    Eg = MAT["Eg"]
    Fc = Eg + quasi_fermi(N, MAT["me"]); Fv = -quasi_fermi(N, MAT["mh"])
    ax = axs[1]
    E = np.linspace(-0.3, 1.1, 500)
    fc = 1 / (1 + np.exp((E - Fc) / KT)); fv = 1 / (1 + np.exp((E - Fv) / KT))
    cb, vb = E >= Eg, E <= 0
    ax.plot(fc[cb], E[cb], color=C_CB, lw=2.5, label="f$_c$(E): electron occupation in CB")
    ax.plot(fv[vb], E[vb], color=C_VB, lw=2.5, label="f$_v$(E): electron occupation in VB")
    ax.plot(fc[~cb], E[~cb], color=C_CB, lw=1, ls=":"); ax.plot(fv[~vb], E[~vb], color=C_VB, lw=1, ls=":")
    ax.text(0.3, 0.8, "gain where f$_c$(E$_2$) > f$_v$(E$_1$)", fontsize=11)
    ax.axhspan(0, Eg, color="0.93"); ax.text(0.02, Eg / 2, "gap", color="0.5")
    ax.axhline(Fc, color=C_CB, ls="--"); ax.axhline(Fv, color=C_VB, ls="--")
    ax.text(1.04, Fc, "E$_{Fc}$", color=C_CB, va="center")
    ax.text(1.04, Fv, "E$_{Fv}$", color=C_VB, va="center")
    ax.set_xlabel("occupation probability"); ax.set_ylabel("E − E$_v$ (eV)")
    ax.set_title(f"InGaAs, N = P = 2×10¹⁸ cm⁻³ (300 K)\n"
                 f"E$_{{Fc}}$ − E$_{{Fv}}$ = {Fc - Fv:.3f} eV > E$_g$ = {Eg} eV")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.25, 0.5))
    save(fig, "m5_band_filling.png")


def fig_gain_spectrum():
    E = np.linspace(0.70, 1.00, 600)
    Ns = [0.5e18, 1.0e18, 1.5e18, 2.0e18, 3.0e18]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(Ns)))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.9))
    ax = axs[0]
    for N, col in zip(Ns, cmap):
        g, dF = gain_spectrum(N, E)
        ax.plot(E, g, color=col, lw=2.2, label=f"N = {N/1e18:.1f}×10¹⁸")
        if dF > MAT["Eg"]:
            ax.plot(dF, 0, "v", color=col, ms=9)
    ax.plot(E, -MAT["alpha0"] * np.sqrt(np.clip(E - MAT["Eg"], 0, None)), "k:", lw=1.5,
            label="N = 0 (absorption)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(MAT["Eg"], color="0.6", ls="--"); ax.text(MAT["Eg"] + 0.003, -2.4e3, "E$_g$", color="0.4")
    ax.set_ylim(-3e3, 2.5e3); ax.set_xlim(E[0], E[-1])
    ax.set_xlabel("photon energy hν (eV)"); ax.set_ylabel("material gain g (cm⁻¹)")
    ax.set_title("Gain spectrum, bulk In$_{0.53}$Ga$_{0.47}$As (model)\n"
                 "▼ = E$_{Fc}$ − E$_{Fv}$ (gain → 0)")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    sec = ax.secondary_xaxis("top", functions=(lambda e: 1.23984 / np.clip(e, 1e-3, None),
                                               lambda l: 1.23984 / np.clip(l, 1e-3, None)))
    sec.set_xlabel("wavelength (µm)")
    # peak gain vs N
    ax = axs[1]
    Nr = np.linspace(0.3e18, 4e18, 60)
    gp = np.array([gain_spectrum(N, E)[0].max() for N in Nr])
    gp_signed = np.array([gain_spectrum(N, E)[0].max() if gain_spectrum(N, E)[0].max() > 1 else
                          gain_spectrum(N, E)[0][np.argmin(abs(E - MAT["Eg"] - 0.005))] for N in Nr])
    ntr = Nr[np.argmax(gp > 5)]
    ax.plot(Nr / 1e18, gp_signed, color=C_CB, lw=2.5, label="model peak gain")
    m = gp > 5
    fit = np.polyfit(Nr[m], gp[m], 1)
    ax.plot(Nr[m] / 1e18, np.polyval(fit, Nr[m]), "--", color=C_PH,
            label=f"linear fit: dg/dN ≈ {fit[0]:.1e} cm²")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(ntr / 1e18, color="0.6", ls="--")
    ax.text(ntr / 1e18 + 0.25, -650, f"N$_{{tr}}$ ≈ {ntr/1e18:.1f}×10¹⁸ cm⁻³\n(transparency)", color="0.3")
    ax.set_xlabel("carrier density N (10¹⁸ cm⁻³)"); ax.set_ylabel("peak gain (cm⁻¹)")
    ax.set_title("Bulk: peak gain ≈ linear in N above N$_{tr}$\n(quantum wells: ≈ g$_0$ ln(N/N$_{tr}$))")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    save(fig, "m5_gain_spectrum.png")

    # data for the interactive OJS slide
    Ngrid = np.round(np.arange(0.2, 4.01, 0.1), 2)
    data = {"E": np.round(E[::3], 4).tolist(), "N": Ngrid.tolist(), "g": [], "dF": []}
    for n in Ngrid:
        g, dF = gain_spectrum(n * 1e18, E)
        data["g"].append(np.round(g[::3], 1).tolist()); data["dF"].append(round(dF, 4))
    with open(os.path.join(HERE, "m5_gain_data.json"), "w") as f:
        json.dump(data, f)
    print("wrote m5_gain_data.json")


# ---------------------------------------------------------------------------
# 5. SOA: power growth along the device and gain saturation
# ---------------------------------------------------------------------------
def soa_propagate(Pin_mW, G0_dB=25, L=1.0, alpha=5.0, Psat=10.0, z_out=None):
    """dP/dz = (Γg0/(1+P/Psat) - α) P ;  L in mm, α in cm^-1, P in mW."""
    a = alpha / 10  # mm^-1
    Gg0 = (G0_dB / 10 * np.log(10)) / L + a
    f = lambda z, P: (Gg0 / (1 + P / Psat) - a) * P
    sol = solve_ivp(f, (0, L), [Pin_mW], t_eval=z_out, rtol=1e-8, atol=1e-12)
    return sol


def fig_soa():
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.9))
    ax = axs[0]
    z = np.linspace(0, 1.0, 200)
    for Pin_dBm, col in zip((-30, -20, -10, 0), plt.cm.plasma(np.linspace(0.1, 0.8, 4))):
        sol = soa_propagate(10 ** (Pin_dBm / 10), z_out=z)
        ax.plot(z, 10 * np.log10(sol.y[0]), color=col, lw=2.4, label=f"P$_{{in}}$ = {Pin_dBm} dBm")
    ax.axhline(10, color="0.5", ls="--"); ax.text(0.02, 10.6, "P$_{sat}$ = 10 dBm", color="0.4")
    ax.set_xlabel("position along SOA, z (mm)"); ax.set_ylabel("signal power (dBm)")
    ax.set_title("Exponential growth, then saturation\n(G$_0$ = 25 dB, L = 1 mm, α = 5 cm⁻¹)")
    ax.legend(frameon=False, fontsize=10)
    ax = axs[1]
    Pins = np.linspace(-40, 10, 120)
    G, Pout = [], []
    for p in Pins:
        P = 10 ** (p / 10)
        out = soa_propagate(P).y[0, -1]
        G.append(10 * np.log10(out / P)); Pout.append(10 * np.log10(out))
    G, Pout = np.array(G), np.array(Pout)
    ax.plot(Pout, G, color=C_CB, lw=2.5)
    i3 = np.argmin(abs(G - (G[0] - 3)))
    ax.plot(Pout[i3], G[i3], "o", color=C_PH, ms=9)
    ax.axhline(G[0] - 3, color="0.6", ls=":")
    ax.annotate(f"P$_{{sat,out}}$ (−3 dB) ≈ {Pout[i3]:.1f} dBm", (Pout[i3], G[i3]),
                (Pout[i3] - 25, G[i3] - 6), arrowprops=dict(arrowstyle="->"))
    ax.text(-15, G[0] + 0.6, "linear regime: G ≈ G$_0$", color="0.3")
    ax.set_xlabel("output power (dBm)"); ax.set_ylabel("gain (dB)")
    ax.set_title("Gain compression: G vs. output power")
    ax.set_ylim(G.min() - 2, G[0] + 3)
    save(fig, "m5_soa_saturation.png")


# ---------------------------------------------------------------------------
# 6. Iso-energy contours and group velocity
# ---------------------------------------------------------------------------
def tb_energy(kx, ky):  # 2D square-lattice tight binding, E in units of t, k in 1/a
    return -2 * (np.cos(kx) + np.cos(ky))


def tb_ray(E, th, n=4000):
    """k on the contour E along direction th (one crossing inside the zone), or None."""
    rmax = np.pi / max(abs(np.cos(th)), abs(np.sin(th)))
    r = np.linspace(0, rmax, n)
    e = tb_energy(r * np.cos(th), r * np.sin(th))
    if E < e[0] or E > e[-1]:
        return None
    rr = np.interp(E, e, r)
    return rr * np.cos(th), rr * np.sin(th)


def fig_isoenergy_tb():
    """Square lattice: contours change from circles to squares to disjoint pockets."""
    fig = plt.figure(figsize=(15, 5.0))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection="polar")
    levels = [(-3.5, "#1f6fb4"), (-2.0, "#2e8b57"), (0.0, "#e67e22"), (2.0, "#c0392b")]
    k = np.linspace(-np.pi, np.pi, 401)
    KX, KY = np.meshgrid(k, k)
    # (a) free electron: circles, v parallel to k, same speed in all directions
    ax = ax0
    for kr, (_, c) in zip([0.9, 1.6, 2.2, 2.7], levels):
        ax.add_patch(plt.Circle((0, 0), kr, fill=False, color=c, lw=2.2))
    for th in np.deg2rad(np.arange(0, 360, 45)):
        x, y = 2.2 * np.cos(th), 2.2 * np.sin(th)
        ax.annotate("", (x + 0.75 * np.cos(th), y + 0.75 * np.sin(th)), (x, y),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax.set_title("Free electron  E = ħ²k²/2m\nround roads, same speed in every direction")
    # (b) crystal: tight-binding square lattice
    ax = ax1
    E = tb_energy(KX, KY)
    ax.contour(KX, KY, E, levels=np.arange(-3.75, 4, 0.5), colors="0.8", linewidths=0.7, linestyles="solid")
    for lev, c in levels:
        ax.contour(KX, KY, E, levels=[lev], colors=c, linewidths=2.4, linestyles="solid")
    for lev, c in levels[1::2]:
        for th in np.deg2rad(np.arange(0, 360, 15)):
            p = tb_ray(lev, th)
            if p is None:
                continue
            vx, vy = 2 * np.sin(p[0]), 2 * np.sin(p[1])  # v = (1/ħ) dE/dk, units t·a/ħ
            ax.annotate("", (p[0] + 0.22 * vx, p[1] + 0.22 * vy), p,
                        arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    ax.text(0, 0, "Γ", ha="center", va="center", fontsize=12)
    ax.text(np.pi * 0.97, -0.25, "X", ha="right", fontsize=12)
    ax.text(np.pi * 0.97, np.pi * 0.93, "M", ha="right", va="top", fontsize=12)
    ax.set_title("Crystal (square lattice)  E = −2t(cos k$_x$a + cos k$_y$a)\n"
                 "circle → square → disjoint pockets")
    for ax in (ax0, ax1):
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi); ax.set_aspect("equal")
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["−π/a", "0", "π/a"])
        ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels(["−π/a", "0", "π/a"])
        ax.set_xlabel("k$_x$"); ax.set_ylabel("k$_y$")
    # (c) speed vs. direction of k
    ax = ax2
    th = np.linspace(0, 2 * np.pi, 721)
    for lev, c in levels:
        sp = []
        for t_ in th:
            p = tb_ray(lev, t_)
            sp.append(np.nan if p is None else 2 * np.hypot(np.sin(p[0]), np.sin(p[1])))
        ax.plot(th, sp, color=c, lw=2.4, label=f"E = {lev:+.1f} t")
    ax.set_title("speed |v| vs. direction of k\n(units of ta/ħ)", pad=18)
    ax.set_rlim(0, 2.9); ax.set_rticks([1, 2])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    save(fig, "m5_isoenergy_tb.png")


def fig_isoenergy_real():
    """EPM iso-energy contours in the k_z = 0 plane: Si CB, GaAs CB, Si top VB."""
    from scipy.interpolate import RegularGridInterpolator
    HBAR, Q = 1.0546e-34, 1.602e-19
    cases = [  # material, band, title, half-width of the k window, contour levels
        ("Si", 4, "Si conduction band:\n4 disjoint ellipses (+2 along ±k$_z$)", 1.0,
         np.array([0.05, 0.15, 0.25, 0.35])),
        ("GaAs", 4, "GaAs conduction band (zoom ×4):\none round valley at Γ", 0.25,
         np.array([0.05, 0.15, 0.25, 0.35])),
        ("Si", 3, "Si top valence band (zoom ×2):\nwarped 'star' contours", 0.5,
         -np.array([0.05, 0.15, 0.25, 0.35])),
    ]
    n = 141
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (mat, b, title, w, levs) in zip(axs, cases):
        k = np.linspace(-w, w, n)
        KX, KY = np.meshgrid(k, k)
        pts = np.c_[KX.ravel(), KY.ravel(), np.zeros(KX.size)]
        Eb = epm_bands(mat, pts).reshape(n, n, -1)
        if b == 4:
            E = Eb[:, :, 4] - Eb[:, :, 4].min()
        else:
            E = Eb[:, :, 3] - epm_bands(mat, np.zeros((1, 3)))[0, 3]
        out = np.abs(KX) + np.abs(KY) > 1.5  # BZ cut at k_z = 0
        Em = np.ma.array(E, mask=out)
        if b == 4:
            ax.contourf(KX, KY, Em, levels=np.arange(0, 0.451, 0.05), cmap="Blues_r", extend="max")
        else:
            ax.contourf(KX, KY, Em, levels=np.arange(-0.45, 0.001, 0.05), cmap="Reds", extend="min")
        cs = ax.contour(KX, KY, Em, levels=np.sort(levs), colors="k", linewidths=1.0,
                        linestyles="solid")
        # group velocity v = (1/ħ) ∇_k E, sampled along the first contour
        a = EPM[mat][0] * 1e-10
        dk = (k[1] - k[0]) * 2 * np.pi / a
        gy, gx = np.gradient(E * Q, dk)
        ivx = RegularGridInterpolator((k, k), gx / HBAR)
        ivy = RegularGridInterpolator((k, k), gy / HBAR)
        lev0 = levs[0] if mat == "GaAs" else levs[1]
        paths = ax.contour(KX, KY, Em, levels=[lev0] if b == 4 else [lev0 - 1e-9, lev0][-1:],
                           colors=C_ACC, linewidths=2.2, linestyles="solid").allsegs[0]
        sc = {("Si", 4): 0.07 / 4e5, ("GaAs", 4): 0.05 / 5e5, ("Si", 3): 0.1 / 3e5}[(mat, b)]
        for seg in paths:
            if len(seg) < 8:
                continue
            for p in seg[:: max(len(seg) // 12, 1)]:
                vx, vy = ivx(p[::-1])[0], ivy(p[::-1])[0]
                ax.annotate("", (p[0] + sc * vx, p[1] + sc * vy), p,
                            arrowprops=dict(arrowstyle="-|>", color=C_ACC, lw=1.4))
        bz = np.array([[1, .5], [.5, 1], [-.5, 1], [-1, .5], [-1, -.5], [-.5, -1],
                       [.5, -1], [1, -.5], [1, .5]])
        ax.plot(bz[:, 0], bz[:, 1], color="0.4", lw=1.2, ls="--")
        if mat == "Si" and b == 4:  # speed along and across the +k_x valley
            j = n // 2 + np.argmin(E[n // 2, n // 2:])  # +k_x valley
            row, col = E[n // 2], E[:, j]
            pl = (k[j + np.argmin(np.abs(row[j:] - lev0))], 0.0)
            pt = (k[j], k[n // 2 + np.argmin(np.abs(col[n // 2:] - lev0))])
            vl, vt = abs(float(ivx([pl[::-1]])[0])), abs(float(ivy([pt[::-1]])[0]))
            ax.annotate(f"along the valley:\n{vl/1e5:.1f}×10⁵ m/s", pl, (0.2, -0.5),
                        fontsize=10, arrowprops=dict(arrowstyle="->", color="0.3"))
            ax.annotate(f"across:\n{vt/1e5:.1f}×10⁵ m/s", pt, (0.25, 0.4), fontsize=10,
                        arrowprops=dict(arrowstyle="->", color="0.3"))
            print(f"Si CB, {lev0*1000:.0f} meV above Ec: v_long = {vl:.3g}, v_trans = {vt:.3g} m/s")
        ax.set_title(title); ax.set_aspect("equal")
        ax.set_xlim(-w * 1.03, w * 1.03); ax.set_ylim(-w * 1.03, w * 1.03)
        ax.set_xlabel("k$_x$ [100]  (2π/a)"); ax.set_ylabel("k$_y$ [010]  (2π/a)")
    fig.text(0.5, -0.01, "Empirical pseudopotential model, k$_z$ = 0 plane. Dashed: Brillouin-zone boundary. "
             "Contours 50, 150, 250, 350 meV from the band edge.\nGreen: one contour (150 meV for Si, 50 meV for GaAs) with its group velocity v = ∇$_k$E/ħ "
             "(arrow length ∝ speed, always perpendicular to the contour).",
             ha="center", fontsize=11, color="0.3")
    fig.tight_layout()
    save(fig, "m5_isoenergy_real.png")


if __name__ == "__main__":
    fig_isoenergy_tb()
    fig_isoenergy_real()
    fig_band_structures()
    fig_direction_cut()
    fig_valleys()
    fig_direct_indirect()
    fig_three_processes()
    fig_einstein_ratio()
    fig_fermi_filling()
    fig_gain_spectrum()
    fig_soa()
