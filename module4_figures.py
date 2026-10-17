"""
Figures for Module 4 - PN junctions, phase shifters, ring and Mach-Zehnder modulators.

Run:  python module4_figures.py            (all figures)
      python module4_figures.py fig_ring_eye fig_mzm   (selected ones)
Writes m4_*.png next to this script. Every figure on the Module 4 slides that
is labelled "script module4_figures.py" comes from here, and the code excerpts
shown on the slides are condensed versions of the functions below.

Physics models (deliberately simple, for teaching):
  * PN junction: abrupt junction, depletion approximation (Poisson).
  * Plasma dispersion: Soref & Bennett (1987) fits at 1310 and 1550 nm.
  * Depletion phase shifter: lateral PN junction in a 500 x 220 nm rib, 1D
    lateral Gaussian mode intensity, first-order overlap with the change of
    carrier density, vertical confinement factor Gamma_y.
  * Carrier mobility vs doping: Masetti/Arora-type fit (for series resistance).
  * Injection phase shifter: rate equation dN/dt = I/(qV) - N/tau.
  * Rings: steady-state transfer functions and time-domain coupled-mode theory
    (CMT) for the large-signal modulator response / eye diagrams.
  * MZM: interference of two arms, chirp from the output-field phase,
    travelling-wave electro-optic response with RF loss and velocity mismatch.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
C1, C2, C3, C4, C5 = "#1f6fb4", "#c0392b", "#2e8b57", "#e67e22", "#7d3c98"
C0 = 299792458.0
RNG = np.random.default_rng(4)

# Silicon constants (cm units, as on the slides)
Q = 1.602e-19            # C
EPS_S = 11.7 * 8.854e-14  # F/cm
VT = 0.02585              # kT/q at 300 K (V)
NI = 1.45e10              # cm^-3 (textbook value used on the slides)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# 1. PN junction (depletion approximation)
# ---------------------------------------------------------------------------
def vbi(NA, ND):
    """Built-in potential (V)."""
    return VT * np.log(NA * ND / NI ** 2)


def depletion(NA, ND, V):
    """Depletion width W and its split x_p, x_n (cm). V > 0 is forward bias."""
    W = np.sqrt(2 * EPS_S / Q * (vbi(NA, ND) - V) * (NA + ND) / (NA * ND))
    return W, W * ND / (NA + ND), W * NA / (NA + ND)


def field_profile(NA, ND, V, x):
    """Triangular field E(x) (V/cm) with the junction at x = 0 (p side x < 0)."""
    W, xp, xn = depletion(NA, ND, V)
    E = np.zeros_like(x)
    m = (x > -xp) & (x <= 0)
    E[m] = -Q * NA / EPS_S * (x[m] + xp)
    m = (x > 0) & (x < xn)
    E[m] = -Q * ND / EPS_S * (xn - x[m])
    return E


def fig_pn_junction():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    NA, ND = 1e17, 3e17
    x = np.linspace(-0.25e-4, 0.15e-4, 2000)
    for V, c in ((0.0, C1), (-2.0, C2)):
        W, xp, xn = depletion(NA, ND, V)
        rho = np.where((x > -xp) & (x < 0), -NA, 0) + np.where((x > 0) & (x < xn), ND, 0)
        ax[0].plot(x * 1e7, rho / 1e17, c=c, lw=2, label=f"V = {V:+.0f} V")
        ax[1].plot(x * 1e7, -field_profile(NA, ND, V, x) / 1e3, c=c, lw=2,
                   label=f"W = {W*1e7:.0f} nm")
    ax[0].set(xlabel="x (nm)", ylabel="ρ/q (10¹⁷ cm⁻³)",
              title=f"Space charge, N_A = 1e17, N_D = 3e17")
    ax[0].legend(frameon=False)
    ax[1].set(xlabel="x (nm)", ylabel="|E| (kV/cm)", title="Field: triangle, area = V_bi − V")
    ax[1].legend(frameon=False)

    V = np.linspace(-4, 0.4, 300)
    h = 0.22e-4                                   # junction height (rib) in cm
    for N, c in ((1e17, C1), (3e17, C3), (1e18, C2)):
        W = depletion(N, N, V)[0]
        Cmm = EPS_S * h / W * 0.1 * 1e15           # fF per mm of length
        ax[2].plot(V, Cmm, c=c, lw=2, label=f"N = {N:.0e} cm⁻³")
    ax[2].set(xlabel="bias V (V)  (V < 0: reverse)", ylabel="C′ (fF/mm)",
              title="Lateral junction capacitance (h = 220 nm)", ylim=(0, 900))
    ax[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "m4_pn_junction.png")


# ---------------------------------------------------------------------------
# 2. Plasma dispersion (Soref & Bennett)
# ---------------------------------------------------------------------------
SOREF = {  # (a_e, a_h, exp_h) for Δn ; (b_e, b_h) for Δα [cm^-1]
    1.55: (8.8e-22, 8.5e-18, 0.8, 8.5e-18, 6.0e-18),
    1.31: (6.2e-22, 6.0e-18, 0.8, 6.0e-18, 4.0e-18),
}


def soref(dNe, dNh, lam=1.55):
    """Index change and absorption change (cm^-1) for carrier changes (cm^-3)."""
    ae, ah, p, be, bh = SOREF[lam]
    return -(ae * dNe + ah * dNh ** p), be * dNe + bh * dNh


def fig_soref():
    N = np.logspace(16, 19, 200)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for lam, ls in ((1.55, "-"), (1.31, "--")):
        ae, ah, p, be, bh = SOREF[lam]
        ax[0].loglog(N, ae * N, ls, c=C1, lw=2, label=f"electrons {lam*1000:.0f} nm")
        ax[0].loglog(N, ah * N ** p, ls, c=C2, lw=2, label=f"holes {lam*1000:.0f} nm")
        ax[1].loglog(N, 10 / np.log(10) * be * N, ls, c=C1, lw=2)
        ax[1].loglog(N, 10 / np.log(10) * bh * N, ls, c=C2, lw=2)
        fom_e = ae * N / (be * N)
        fom_h = ah * N ** p / (bh * N)
        ax[2].semilogx(N, fom_e * 1e4, ls, c=C1, lw=2)
        ax[2].semilogx(N, fom_h * 1e4, ls, c=C2, lw=2)
    ax[0].set(xlabel="ΔN (cm⁻³)", ylabel="|Δn|", title="Index change")
    ax[0].legend(frameon=False, fontsize=10)
    ax[1].set(xlabel="ΔN (cm⁻³)", ylabel="Δα (dB/cm)", title="Free-carrier absorption")
    ax[2].set(xlabel="ΔN (cm⁻³)", ylabel="|Δn|/Δα  (µm)",
              title="Figure of merit: holes win below ~10¹⁹")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "m4_soref.png")


# ---------------------------------------------------------------------------
# 3. Depletion phase shifter (lateral PN junction in a rib)
# ---------------------------------------------------------------------------
def mobility(N, carrier="e"):
    """Low-field mobility vs doping (cm^2/Vs), Arora/Masetti-type fit."""
    if carrier == "e":
        mn, mx, Nr, a = 52.2, 1417.0, 9.68e16, 0.68
    else:
        mn, mx, Nr, a = 44.9, 470.5, 2.23e17, 0.719
    return mn + (mx - mn) / (1 + (N / Nr) ** a)


def depletion_ps(NA, ND, V, x0=0.0, lam=1.55, w0=0.28e-4, gamma_y=0.8):
    """First-order Δn_eff and loss of a lateral PN phase shifter.

    Junction at x = x0 (cm) inside the rib, p side on the left. The lateral
    mode intensity is a Gaussian of 1/e^2 half-width w0, so the overlap with
    a uniform slab of carriers between x1 and x2 is an erf difference.
    Returns (Δn_eff relative to V = 0, loss in dB/cm at bias V).
    """
    ae, ah, p, be, bh = SOREF[lam]
    ov = lambda x1, x2: 0.5 * (erf(np.sqrt(2) * x2 / w0) - erf(np.sqrt(2) * x1 / w0))
    _, xp0, xn0 = depletion(NA, ND, 0.0)
    _, xp, xn = depletion(NA, ND, V)
    # carriers removed between the 0 V and the V depletion edges -> index rises
    dneff = gamma_y * (ae * ND * ov(x0 + xn0, x0 + xn) + ah * NA ** p * ov(x0 - xp, x0 - xp0))
    alpha = gamma_y * (be * ND * ov(x0 + xn, np.inf) + bh * NA * ov(-np.inf, x0 - xp))
    return dneff, 10 / np.log(10) * alpha


def ps_metrics(N, Vswing=2.0, lam=1.55, d_contact=1.0e-4, h_slab=0.09e-4, h=0.22e-4):
    """V_pi*L (V cm), loss (dB/cm), C' (fF/mm), R'L (ohm cm) and f_RC (GHz)."""
    dneff, _ = depletion_ps(N, N, -Vswing, lam=lam)
    VpiL = Vswing * lam * 1e-4 / (2 * dneff)
    loss = depletion_ps(N, N, -Vswing / 2, lam=lam)[1]
    W = depletion(N, N, -Vswing / 2)[0]
    Cp = EPS_S * h / W                              # F/cm
    rho = 1 / (Q * mobility(N, "e") * N) + 1 / (Q * mobility(N, "h") * N)
    RL = rho * d_contact / h_slab                   # ohm*cm (both sides, 1 µm slabs)
    return VpiL, loss, Cp * 0.1 * 1e15, RL, 1 / (2 * np.pi * RL * Cp) / 1e9


def fig_depletion_ps():
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    # (a) geometry: mode and depleted region
    N = 3e17
    x = np.linspace(-0.6, 0.6, 1000)
    ax[0].fill_between(x, np.exp(-2 * x ** 2 / 0.28 ** 2), color="0.85", label="mode |E|²")
    for V, c in ((0, C1), (-2, C2)):
        W, xp, xn = depletion(N, N, V)
        ax[0].axvspan(-xp * 1e4, xn * 1e4, color=c, alpha=0.25, label=f"depleted, V = {V} V")
    ax[0].axvspan(-0.25, -0.6, color=C2, alpha=0.05)
    ax[0].text(-0.45, 0.9, "P", fontsize=16, ha="center")
    ax[0].text(0.45, 0.9, "N", fontsize=16, ha="center")
    ax[0].set(xlabel="x (µm)", yticks=[], title="Lateral PN, N = 3×10¹⁷ cm⁻³")
    ax[0].legend(frameon=False, fontsize=10, loc="lower center")
    # (b) Δn_eff and phase vs V
    V = np.linspace(0, -4, 41)
    for N, c in ((1e17, C1), (3e17, C3), (1e18, C2)):
        dn = np.array([depletion_ps(N, N, v)[0] for v in V])
        ax[1].plot(-V, dn * 1e5, c=c, lw=2, label=f"N = {N:.0e}")
    ax[1].set(xlabel="reverse bias (V)", ylabel="Δn_eff (×10⁻⁵)",
              title="Index change: sub-linear in V")
    ax[1].legend(frameon=False)
    # (c) doping trade-off
    Ns = np.logspace(16.7, 18.5, 30)
    m = np.array([ps_metrics(n) for n in Ns])
    ax[2].semilogx(Ns, m[:, 0], c=C1, lw=2.5, label="V_πL (V·cm)")
    ax[2].semilogx(Ns, m[:, 1] / 10, c=C2, lw=2.5, label="loss (×10 dB/cm)")
    ax[2].semilogx(Ns, m[:, 2] / 100, c=C3, lw=2.5, label="C′ (×100 fF/mm)")
    ax[2].set(xlabel="doping N_A = N_D (cm⁻³)", ylim=(0, 6),
              title="Doping ↑: V_πL ↓, loss ↑, C ↑")
    ax[2].legend(frameon=False, fontsize=11)
    ax[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "m4_depletion_ps.png")


# ---------------------------------------------------------------------------
# 4. Injection (PIN) phase shifter
# ---------------------------------------------------------------------------
def injection_static(I, L=0.05, w=0.5e-4, h=0.22e-4, tau=1e-9, gamma=0.8, lam=1.55):
    """Steady state: ΔN = I tau / (q w h L), Δφ (rad) and extra loss (dB)."""
    dN = I * tau / (Q * w * h * L)
    dn, da = soref(dN, dN, lam)
    phi = 2 * np.pi / (lam * 1e-4) * gamma * abs(dn) * L
    return dN, phi, 10 / np.log(10) * gamma * da * L


def injection_dynamics(bits, Tb, tau=1e-9, sps=64, pre=0.0, tpre=0.15e-9):
    """Normalised ΔN(t) for an NRZ current drive with optional pre-emphasis."""
    dt = Tb / sps
    drive = np.repeat(bits.astype(float), sps)
    if pre:
        kick = np.zeros_like(drive)
        n = int(tpre / dt)
        for k in range(1, len(bits)):
            if bits[k] != bits[k - 1]:
                kick[k * sps:k * sps + n] = pre * (1 if bits[k] else -1)
        drive = drive + kick
    n = np.zeros_like(drive)                       # ΔN / ΔN_steady-state
    for i in range(1, len(drive)):
        n[i] = max(n[i - 1] + dt * (drive[i - 1] - n[i - 1]) / tau, 0.0)
    return np.arange(len(drive)) * dt, drive, n


def fig_injection():
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    I = np.linspace(0, 12e-3, 200)
    dN, phi, loss = injection_static(I)
    ax[0].plot(I * 1e3, phi / np.pi, c=C1, lw=2.5, label="Δφ/π")
    ax[0].plot(I * 1e3, loss, c=C2, lw=2.5, label="excess loss (dB)")
    Ipi = I[np.argmin(abs(phi - np.pi))]
    ax[0].axvline(Ipi * 1e3, c="k", ls=":")
    ax[0].text(Ipi * 1e3 + 0.2, 1.6, f"I_π ≈ {Ipi*1e3:.1f} mA\nΔN ≈ {injection_static(Ipi)[0]:.1e}",
               fontsize=11)
    ax[0].set(xlabel="forward current (mA)", title="PIN, L = 500 µm, τ = 1 ns, both carriers")
    ax[0].legend(frameon=False)
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0])
    Tb = 0.2e-9                                     # 5 Gb/s
    t, d, N = injection_dynamics(bits, Tb)
    t2, d2, N2 = injection_dynamics(bits, Tb, pre=4.0)
    ax[1].plot(t * 1e9, d, c="0.6", lw=1, label="drive (NRZ)")
    ax[1].plot(t * 1e9, N, c=C2, lw=2, label="ΔN, plain drive")
    ax[1].plot(t2 * 1e9, N2, c=C1, lw=2, label="ΔN, pre-emphasis")
    ax[1].set(xlabel="time (ns)", ylabel="normalised", ylim=(-0.1, 1.4),
              title="5 Gb/s: carrier lifetime (1 ns) ≫ bit (200 ps)")
    ax[1].legend(frameon=False, fontsize=10, ncol=3, loc="upper center")
    fig.tight_layout()
    save(fig, "m4_injection.png")


# ---------------------------------------------------------------------------
# 5. Thermo-optic phase shifter
# ---------------------------------------------------------------------------
def fig_thermal():
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    lam, dndT = 1.55e-6, 1.86e-4
    L = np.linspace(50e-6, 1e-3, 200)
    dT = lam / (2 * L * dndT)
    for RthL, c, lab in ((0.2, C2, "standard (R_th·L = 0.2 K·m/W)"),
                         (2.0, C1, "with undercut (R_th·L = 2 K·m/W)")):
        P = dT / (RthL / L)
        ax[0].plot(L * 1e6, P * 1e3, c=c, lw=2.5, label=lab)
    ax[0].set(xlabel="heater length L (µm)", ylabel="P_π (mW)", ylim=(0, 30),
              title="P_π = ΔT_π/R_th is independent of L")
    ax2 = ax[0].twinx()
    ax2.plot(L * 1e6, dT, c="0.5", ls="--", lw=1.5)
    ax2.set_ylabel("ΔT_π (K)  (dashed)", color="0.4")
    ax2.spines["top"].set_visible(False)
    ax[0].legend(frameon=False, fontsize=10, loc="upper right")
    t = np.linspace(0, 200e-6, 1000)
    for tau, c, lab in ((5e-6, C2, "standard, τ ≈ 5 µs"), (50e-6, C1, "undercut, τ ≈ 50 µs")):
        on = (t < 100e-6)
        y = np.where(on, 1 - np.exp(-t / tau), (1 - np.exp(-100e-6 / tau)) * np.exp(-(t - 100e-6) / tau))
        ax[1].plot(t * 1e6, y, c=c, lw=2.5, label=lab)
    ax[1].set(xlabel="time (µs)", ylabel="ΔT / ΔT_final", title="Step response (heater on for 100 µs)")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "m4_thermal.png")


# ---------------------------------------------------------------------------
# 6. Ring resonators: steady state
# ---------------------------------------------------------------------------
def ring_through(phi, a, t):
    """All-pass ring power transmission (a: round-trip amplitude, t: self-coupling)."""
    return (a ** 2 - 2 * a * t * np.cos(phi) + t ** 2) / (1 - 2 * a * t * np.cos(phi) + (a * t) ** 2)


def ring_adddrop(phi, a, t1, t2):
    """Add-drop ring: (through, drop) power transmissions."""
    den = 1 - 2 * a * t1 * t2 * np.cos(phi) + (a * t1 * t2) ** 2
    thr = (t2 ** 2 * a ** 2 - 2 * a * t1 * t2 * np.cos(phi) + t1 ** 2) / den
    drop = (1 - t1 ** 2) * (1 - t2 ** 2) * a / den
    return thr, drop


def ring_phase(lam, lam0, R, ng=4.2):
    """Round-trip phase relative to resonance at lam0 (group-index model)."""
    Lr = 2 * np.pi * R
    return 2 * np.pi * ng * Lr * (1 / lam - 1 / lam0)


def fig_ring_coupling():
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    R, loss_db = 7.5e-6, 10.0                     # 10 dB/cm (doped ring)
    Lr = 2 * np.pi * R
    a = 10 ** (-loss_db * Lr * 100 / 20)
    lam = np.linspace(1549.6e-9, 1550.4e-9, 3000)
    phi = ring_phase(lam, 1550e-9, R)
    for t, c, lab in ((np.sqrt(1 - 0.3 * (1 - a ** 2)), C1, "under"),
                      (a, C3, "critical"),
                      (np.sqrt(1 - 3 * (1 - a ** 2)), C2, "over")):
        T = ring_through(phi, a, t)
        ax[0].plot((lam - 1550e-9) * 1e12, 10 * np.log10(T), c=c, lw=2,
                   label=f"{lab}, κ² = {1-t**2:.4f}")
    ax[0].set(xlabel="detuning (pm)", ylabel="through (dB)", ylim=(-35, 1),
              title=f"R = 7.5 µm, 10 dB/cm → 1−a² = {1-a**2:.4f}")
    ax[0].legend(frameon=False, fontsize=10)
    # Q and extinction vs coupling
    k2 = np.logspace(-3.3, -0.7, 200)
    t = np.sqrt(1 - k2)
    ng, lam0 = 4.2, 1550e-9
    QL = np.pi * ng * Lr * np.sqrt(a * t) / (lam0 * (1 - a * t))
    Tmin = (a - t) ** 2 / (1 - a * t) ** 2
    ax[1].loglog(k2, QL, c=C1, lw=2.5)
    ax[1].axvline(1 - a ** 2, c=C3, ls=":")
    ax[1].set(xlabel="power coupling κ²", ylabel="loaded Q", title="Q_L falls as coupling grows")
    ax3 = ax[1].twinx()
    ax3.semilogx(k2, -10 * np.log10(Tmin + 1e-6), c=C2, lw=2)
    ax3.set_ylabel("extinction at resonance (dB)", color=C2)
    ax3.spines["top"].set_visible(False)
    # phase response
    for t, c in ((np.sqrt(1 - 0.3 * (1 - a ** 2)), C1), (np.sqrt(1 - 3 * (1 - a ** 2)), C2)):
        H = (t - a * np.exp(-1j * phi)) / (1 - a * t * np.exp(-1j * phi))
        ax[2].plot((lam - 1550e-9) * 1e12, np.unwrap(np.angle(H)) / np.pi, c=c, lw=2)
    ax[2].set(xlabel="detuning (pm)", ylabel="phase (π)",
              title="Phase tells under (blue) from over (red)")
    fig.tight_layout()
    save(fig, "m4_ring_coupling.png")


def fig_ring_modulation():
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    R, loss_db, eta = 7.5e-6, 10.0, 40e-12         # 40 pm/V
    Lr = 2 * np.pi * R
    a = 10 ** (-loss_db * Lr * 100 / 20)
    t = np.sqrt(1 - 1.5 * (1 - a ** 2))             # slightly over-coupled
    lam = np.linspace(1549.6e-9, 1550.4e-9, 3000)
    for V, c in ((0, C1), (1, C3), (2, C2)):
        T = ring_through(ring_phase(lam, 1550e-9 + eta * V, R), a, t)
        ax[0].plot((lam - 1550e-9) * 1e12, T, c=c, lw=2, label=f"V_rev = {V} V")
    det = 0.0
    ax[0].axvline(det * 1e12, c="k", ls="--")
    ax[0].text(det * 1e12 - 8, 0.5, "laser\n(max OMA)", ha="right")
    ax[0].set(xlabel="wavelength − 1550 nm (pm)", ylabel="through (linear)",
              title="Reverse bias red-shifts the resonance (40 pm/V)")
    ax[0].legend(frameon=False, loc="lower right")
    # ER, IL and OMA vs laser detuning for a 2 Vpp swing
    d = np.linspace(-300e-12, 150e-12, 400)
    T0 = ring_through(ring_phase(1550e-9 + d, 1550e-9, R), a, t)
    T1 = ring_through(ring_phase(1550e-9 + d, 1550e-9 + 2 * eta, R), a, t)
    hi, lo = np.maximum(T0, T1), np.minimum(T0, T1)
    ax[1].plot(d * 1e12, 10 * np.log10(hi / lo), c=C1, lw=2.5, label="dynamic ER (dB)")
    ax[1].plot(d * 1e12, -10 * np.log10(hi), c=C2, lw=2.5, label="IL of the '1' level (dB)")
    ax[1].plot(d * 1e12, 10 * (hi - lo), c=C3, lw=2.5, label="OMA (×10, linear)")
    ax[1].set(xlabel="laser detuning from resonance at 0 V (pm)", ylim=(0, 15),
              title="2 Vpp swing: choosing the laser detuning")
    ax[1].legend(frameon=False, fontsize=11)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "m4_ring_modulation.png")


# ---------------------------------------------------------------------------
# 7. Ring modulator dynamics: time-domain coupled-mode theory
# ---------------------------------------------------------------------------
def ring_cmt(v, dt, Qi, Qe, eta, lam0=1550e-9, det=0.0):
    """Large-signal ring modulator response.

    v   : reverse voltage samples (V), the resonance shifts by eta*v (m)
    det : laser wavelength minus resonance at 0 V (m)
    Field a (energy-normalised) obeys
        da/dt = (j(ω0(t) − ωL) − 1/τ) a − j μ s,   s_out = s − j μ a
    with 1/τ = ω0/(2Qi) + ω0/(2Qe) and μ² = ω0/Qe. Exact update per step.
    """
    w0 = 2 * np.pi * C0 / lam0
    ti, te = 2 * Qi / w0, 2 * Qe / w0
    mu = np.sqrt(2 / te)
    wl = 2 * np.pi * C0 / (lam0 + det)
    a, out = 0.0 + 0j, np.empty(len(v))
    # start from the steady state of the first sample
    D = 2 * np.pi * C0 / (lam0 + eta * v[0]) - wl
    a = -1j * mu / (1 / ti + 1 / te - 1j * D)
    for i, vi in enumerate(v):
        D = 2 * np.pi * C0 / (lam0 + eta * vi) - wl
        p = 1j * D - (1 / ti + 1 / te)
        e = np.exp(p * dt)
        a = a * e + (-1j * mu) * (e - 1) / p
        out[i] = abs(1 - 1j * mu * a) ** 2
    return out


def nrz(bits, sps, f3db, dt):
    """NRZ waveform through a first-order RC low-pass."""
    x = np.repeat(bits.astype(float), sps)
    al = 1 - np.exp(-2 * np.pi * f3db * dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + al * (x[i] - y[i - 1])
    return y


def best_detuning(Qi, Qe, eta, Vpp, lam0=1550e-9):
    """Blue-side laser detuning maximising the static OMA."""
    w0 = 2 * np.pi * C0 / lam0
    d = np.linspace(-4, 0.5, 400) * lam0 / (Qi * Qe / (Qi + Qe))

    def T(shift):
        D = 2 * np.pi * C0 * (1 / (lam0 + shift) - 1 / (lam0 + d))
        g = w0 / (2 * Qi) + w0 / (2 * Qe)
        return abs(1 - (w0 / Qe) / (g - 1j * D)) ** 2
    oma = abs(T(eta * Vpp) - T(0.0))
    return d[np.argmax(oma)]


def fig_ring_eye():
    br, sps = 25e9, 32
    dt = 1 / br / sps
    bits = RNG.integers(0, 2, 400)
    Qi, eta, Vpp = 40e3, 40e-12, 2.0
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    Qs = np.logspace(3, 5, 200)
    f_ph = C0 / (1550e-9 * Qs) / 1e9
    f_rc = 1 / (2 * np.pi * 100 * 15e-15) / 1e9
    ax[0].loglog(Qs, f_ph, c=C1, lw=2.5, label="photon lifetime c/(λQ)")
    ax[0].axhline(f_rc, c=C2, lw=2.5, label="RC: 100 Ω × 15 fF")
    ax[0].loglog(Qs, 1 / np.sqrt(1 / f_ph ** 2 + 1 / f_rc ** 2), "k--", lw=1.5, label="combined")
    ax[0].set(xlabel="loaded Q", ylabel="bandwidth (GHz)", title="Bandwidth budget")
    ax[0].legend(frameon=False, fontsize=10)
    ax[0].grid(alpha=0.3, which="both")
    v = Vpp * nrz(bits, sps, 50e9, dt)
    for k, Qe in enumerate((6e3, 12e3, 40e3)):
        QL = Qi * Qe / (Qi + Qe)
        det = best_detuning(Qi, Qe, eta, Vpp)
        P = ring_cmt(v, dt, Qi, Qe, eta, det=det)
        n2 = 2 * sps
        for s in range(8, len(bits) - 2):
            ax[k + 1].plot(np.arange(n2) / sps, P[s * sps - sps // 2:s * sps - sps // 2 + n2],
                           c=C1, lw=0.6, alpha=0.35)
        ax[k + 1].set(xlabel="time (UI)", ylim=(0, 1.05),
                      title=f"Q_L ≈ {QL/1e3:.1f}k  (FWHM {1550e-9/QL*1e12:.0f} pm)")
        if k == 0:
            ax[k + 1].set_ylabel("through power (norm.)")
    fig.suptitle("25 Gb/s NRZ, 2 Vpp, 40 pm/V, Q_i = 40k: high Q = large swing but ISI; low Q = fast but small swing",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    save(fig, "m4_ring_eye.png")


# ---------------------------------------------------------------------------
# 8. PAM4 with a ring: non-linear transfer and pre-distortion
# ---------------------------------------------------------------------------
def fig_pam4():
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    Qi, Qe, eta, Vpp = 40e3, 12e3, 40e-12, 2.0
    det = best_detuning(Qi, Qe, eta, Vpp)
    V = np.linspace(-0.3, 2.3, 400)
    w0 = 2 * np.pi * C0 / 1550e-9

    def T(v):
        D = 2 * np.pi * C0 * (1 / (1550e-9 + eta * v) - 1 / (1550e-9 + det))
        g = w0 / (2 * Qi) + w0 / (2 * Qe)
        return abs(1 - (w0 / Qe) / (g - 1j * D)) ** 2
    TV = np.array([T(v) for v in V])
    ax[0].plot(V, TV, c="k", lw=2)
    lin = np.array([0, 1, 2, 3]) * Vpp / 3
    Tl = np.array([T(v) for v in lin])
    target = np.linspace(Tl[0], Tl[-1], 4)
    m = (V >= 0) & (V <= Vpp)
    pre = np.interp(target, TV[m], V[m]) if TV[m][-1] > TV[m][0] else np.interp(target, TV[m][::-1], V[m][::-1])
    for v, tt in zip(lin, Tl):
        ax[0].plot([v, v, -0.3], [0, tt, tt], c=C2, lw=1, ls="--")
    for v, tt in zip(pre, target):
        ax[0].plot([v, v, 2.3], [0, tt, tt], c=C1, lw=1, ls=":")
    ax[0].set(xlabel="reverse voltage (V)", ylabel="through power", xlim=(-0.3, 2.3), ylim=(0, 1),
              title="Lorentzian slope: equal ΔV ≠ equal ΔP")
    ax[0].text(0.05, 0.92, "red: equally spaced V\nblue: pre-distorted V", transform=ax[0].transAxes, fontsize=11)
    eyes_l, eyes_p = np.diff(Tl), np.diff(target)
    xs = np.arange(3)
    ax[1].bar(xs - 0.18, eyes_l / eyes_l.max(), 0.35, color=C2, label=f"linear drive, RLM = {3*eyes_l.min()/(Tl[-1]-Tl[0]):.2f}")
    ax[1].bar(xs + 0.18, eyes_p / eyes_l.max(), 0.35, color=C1, label="pre-distorted, RLM = 1.00")
    ax[1].set(xticks=xs, xticklabels=["lower eye", "middle eye", "upper eye"],
              ylabel="eye height (norm.)", title="PAM4 level mismatch", ylim=(0, 1.35))
    ax[1].legend(frameon=False, loc="upper center", ncol=2, fontsize=11)
    fig.tight_layout()
    save(fig, "m4_pam4.png")
    return pre


# ---------------------------------------------------------------------------
# 9. Vernier rings
# ---------------------------------------------------------------------------
def fig_vernier():
    fig, ax = plt.subplots(2, 1, figsize=(15, 5.6), sharex=True)
    lam = np.linspace(1480e-9, 1690e-9, 40000)
    a = 0.999
    t = np.sqrt(1 - 0.05)
    out = []
    for R, c in ((10e-6, C1), (10.8e-6, C2)):
        _, D = ring_adddrop(ring_phase(lam, 1550e-9, R), a, t, t)
        out.append(D)
        ax[0].plot(lam * 1e9, D, c=c, lw=1, label=f"R = {R*1e6:.1f} µm, FSR = {1550e-9**2/(4.2*2*np.pi*R)*1e9:.2f} nm")
    ax[0].set(ylabel="drop", title="Two rings with slightly different FSR")
    ax[0].legend(frameon=False, loc="upper right", fontsize=10)
    f1, f2 = (1550e-9 ** 2 / (4.2 * 2 * np.pi * R) for R in (10e-6, 10.8e-6))
    ax[1].plot(lam * 1e9, out[0] * out[1], c="k", lw=1.2)
    ax[1].set(xlabel="wavelength (nm)", ylabel="drop₁ × drop₂",
              title=f"Cascade: FSR_V = FSR₁·FSR₂/|FSR₁−FSR₂| = {f1*f2/abs(f1-f2)*1e9:.0f} nm")
    fig.tight_layout()
    save(fig, "m4_vernier.png")


# ---------------------------------------------------------------------------
# 10. Mach-Zehnder modulator
# ---------------------------------------------------------------------------
def mzm_field(phi1, phi2):
    return 0.5 * (np.exp(1j * phi1) + np.exp(1j * phi2))


def fig_mzm():
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    dphi = np.linspace(0, 2 * np.pi, 400)
    ax[0].plot(dphi / np.pi, np.cos(dphi / 2) ** 2, c="k", lw=2, label="intensity")
    ax[0].plot(dphi / np.pi, np.cos(dphi / 2), c=C5, lw=1.5, ls="--", label="field")
    for x, lab in ((0, "peak"), (0.5, "quadrature\n(OOK/PAM)"), (1, "null\n(BPSK, IQ)")):
        ax[0].plot(x, np.cos(x * np.pi / 2) ** 2, "o", c=C2, ms=9)
        ax[0].annotate(lab, (x, np.cos(x * np.pi / 2) ** 2), (x + 0.08, np.cos(x * np.pi / 2) ** 2 + 0.12), fontsize=11)
    ax[0].set(xlabel="Δφ (π)", ylabel="output (norm.)", ylim=(-1.05, 1.3), title="Transfer function and bias points")
    ax[0].legend(frameon=False, loc="lower left")
    # dynamic ER and OMA vs swing at quadrature
    r = np.linspace(0.05, 1.0, 200)                      # Vpp/Vπ (differential)
    hi = np.cos((np.pi / 2 - r * np.pi / 2) / 2) ** 2
    lo = np.cos((np.pi / 2 + r * np.pi / 2) / 2) ** 2
    ax[1].plot(r, 10 * np.log10(hi / lo), c=C1, lw=2.5, label="dynamic ER (dB)")
    ax[1].plot(r, 10 * (hi - lo), c=C3, lw=2.5, label="OMA (×10, P_in = 1)")
    ax[1].plot(r, -10 * np.log10(hi), c=C2, lw=2.5, label="loss of the '1' level (dB)")
    ax[1].axvspan(0.2, 0.5, color="0.9")
    ax[1].text(0.22, 12, "typical Si\nMZM", fontsize=11)
    ax[1].set(xlabel="V_pp / V_π", ylim=(0, 15), title="Biased at quadrature")
    ax[1].legend(frameon=False, fontsize=11)
    ax[1].grid(alpha=0.3)
    # chirp: single-arm vs push-pull for one rising edge
    t = np.linspace(-60e-12, 60e-12, 1000)
    edge = 0.5 * (1 + np.tanh(t / 8e-12))
    r0 = 0.5
    for mode, c in (("single arm", C2), ("push-pull", C1), ("push-pull 60/40", C4)):
        if mode == "single arm":
            p1, p2 = np.pi / 2 + r0 * np.pi * (edge - 0.5), 0 * t
        elif mode == "push-pull":
            p1, p2 = np.pi / 4 + r0 * np.pi / 2 * (edge - 0.5), -np.pi / 4 - r0 * np.pi / 2 * (edge - 0.5)
        else:
            p1, p2 = np.pi / 4 + 0.6 * r0 * np.pi * (edge - 0.5), -np.pi / 4 - 0.4 * r0 * np.pi * (edge - 0.5)
        E = mzm_field(-p1, -p2)
        f = np.gradient(np.unwrap(np.angle(E)), t) / (2 * np.pi)
        ax[2].plot(t * 1e12, f / 1e9, c=c, lw=2.5, label=mode)
    ax[2].set(xlabel="time (ps)", ylabel="instantaneous Δf (GHz)", title="Chirp on a rising edge")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "m4_mzm.png")


def tw_response(f, L, n_rf=3.8, ng=4.0, a0_db=3.0, f_j=60e9):
    """Travelling-wave E-O response |H(f)| (matched, terminated line).

    a0_db: RF loss coefficient in dB/(cm·sqrt(GHz)); L in m.
    f_j  : intrinsic junction pole 1/(2π R'C') of the loaded PN segment.
    """
    alpha = a0_db * np.sqrt(f / 1e9) * 100 / 8.686     # Np/m
    db = 2 * np.pi * f * (n_rf - ng) / C0                # rad/m
    g = (alpha + 1j * db) * L
    H = np.where(np.abs(g) < 1e-9, 1.0, (1 - np.exp(-g)) / np.where(np.abs(g) < 1e-9, 1, g))
    return np.abs(H) / np.sqrt(1 + (f / f_j) ** 2)


def f3db_tw(L, **kw):
    f = np.linspace(0.1e9, 200e9, 4000)
    H = 20 * np.log10(tw_response(f, L, **kw))
    # electro-optic 3 dB = 6 dB electrical convention not used: 3 dB of |H|^2 -> -3 dB in 10log
    i = np.argmax(10 * np.log10(tw_response(f, L, **kw) ** 2) < -3)
    return f[i] if i > 0 else f[-1]


def fig_tw_mzm():
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    f = np.linspace(0.1e9, 100e9, 1000)
    for L, c in ((1e-3, C1), (2e-3, C3), (3e-3, C4), (4e-3, C2)):
        ax[0].plot(f / 1e9, 10 * np.log10(tw_response(f, L) ** 2), c=c, lw=2.5, label=f"L = {L*1e3:.0f} mm")
    ax[0].plot(f / 1e9, 10 * np.log10(tw_response(f, 3e-3, n_rf=2.6) ** 2), c=C4, lw=1.5, ls="--",
               label="3 mm, unloaded line n_RF = 2.6")
    ax[0].axhline(-3, c="k", ls=":")
    ax[0].set(xlabel="frequency (GHz)", ylabel="E-O response (dB)", ylim=(-12, 0.5),
              title="TW-MZM: 3 dB/(cm·√GHz), n_RF 3.8, n_g 4.0, f_j 60 GHz")
    ax[0].legend(frameon=False, fontsize=10)
    Ls = np.linspace(0.5e-3, 6e-3, 40)
    bw = np.array([f3db_tw(L) for L in Ls]) / 1e9
    VpiL = 2.0                                       # V·cm, single arm
    ax[1].plot(Ls * 1e3, bw, c=C1, lw=2.5, label="3-dB bandwidth (GHz)")
    ax[1].plot(Ls * 1e3, VpiL / (Ls * 100) * 10, c=C2, lw=2.5, label="V_π per arm (×0.1 V)")
    ax[1].set(xlabel="phase-shifter length (mm)", ylim=(0, 120),
              title="Length trade-off (V_πL = 2 V·cm)")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "m4_tw_mzm.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    fig_pn_junction()
    fig_soref()
    fig_depletion_ps()
    fig_injection()
    fig_thermal()
    fig_ring_coupling()
    fig_ring_modulation()
    fig_ring_eye()
    fig_pam4()
    fig_vernier()
    fig_mzm()
    fig_tw_mzm()
