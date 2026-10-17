"""
Figures for Module 6 - compact models, PDK/yield, microwave photonics, layout.

Run:  python module6_figures.py
Writes m6_*.png next to this script. Every figure on the Module 6 slides that
is labelled "script module6_figures.py" comes from here, and the code excerpts
shown on the slides are condensed versions of the functions below.

Physics models (deliberately simple, for teaching):
  * Waveguides: effective index method (EIM) built on a multilayer slab
    solver (transfer matrix, TE/TM). Si and SiO2 dispersion from Sellmeier.
  * Circuits: generic S-matrix netlist solver (the same algorithm used by
    circuit simulators such as IPKISS/Caphe, Interconnect, SAX).
  * Analog links: MZM biased at quadrature, Bessel expansion for harmonics
    and intermodulation, shot/thermal/RIN noise (Cox, "Analog optical links").
  * Thermal: 2D finite-volume steady-state heat equation on the SOI stack.
  * Bend loss: Marcuse formula for a bent slab, applied to the EIM lateral
    index profile.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from scipy.optimize import brentq
from scipy.special import jv
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy import sparse
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
C1, C2, C3, C4, C5 = "#1f6fb4", "#c0392b", "#2e8b57", "#e67e22", "#7d3c98"
C0 = 299792458.0
RNG = np.random.default_rng(6)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# 1. Materials and the multilayer slab solver
# ---------------------------------------------------------------------------
def n_si(lam_um):
    """Si refractive index, Li (1980) Sellmeier, room temperature."""
    l2 = lam_um ** 2
    return np.sqrt(11.6858 + 0.939816 / l2 + 0.00810461 * 1.1071 ** 2 / (l2 - 1.1071 ** 2))


def n_sio2(lam_um):
    """Fused silica, Malitson (1965)."""
    l2 = lam_um ** 2
    return np.sqrt(1 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
                   + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                   + 0.8974794 * l2 / (l2 - 9.896161 ** 2))


def slab_modes(ns, ds, lam, pol="TE"):
    """Guided-mode effective indices of a multilayer slab.

    ns : indices [n_clad_left, n_1, ..., n_k, n_clad_right]
    ds : thicknesses of the k inner layers (same unit as lam)
    Returns neff of all guided modes, highest first.
    """
    ns = np.asarray(ns, float)
    k0 = 2 * np.pi / lam
    p = (lambda n: 1.0) if pol == "TE" else (lambda n: 1.0 / n ** 2)

    def f(neff):
        g_l = k0 * np.sqrt(neff ** 2 - ns[0] ** 2)
        g_r = k0 * np.sqrt(neff ** 2 - ns[-1] ** 2)
        v = np.array([1.0 + 0j, p(ns[0]) * g_l])          # (psi, p psi')
        for n, d in zip(ns[1:-1], ds):
            kx = k0 * np.lib.scimath.sqrt(n ** 2 - neff ** 2)
            q = p(n) * kx
            c, s = np.cos(kx * d), np.sin(kx * d)
            sq = s / q if abs(q) > 1e-12 else d / p(n)
            v = np.array([c * v[0] + sq * v[1], -q * s * v[0] + c * v[1]])
        return (v[1] + p(ns[-1]) * g_r * v[0]).real

    lo, hi = max(ns[0], ns[-1]) + 1e-7, ns.max() - 1e-7
    grid = np.linspace(lo, hi, 400)
    vals = np.array([f(x) for x in grid])
    roots = [brentq(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1)
             if np.sign(vals[i]) != np.sign(vals[i + 1])]
    return sorted(roots, reverse=True)


def neff_strip(w, h=0.22, lam=1.55):
    """TE-like fundamental mode of a Si strip in SiO2 (EIM). Units: um."""
    nsi, nox = n_si(lam), n_sio2(lam)
    n_slab = slab_modes([nox, nsi, nox], [h], lam, "TE")[0]
    return slab_modes([nox, n_slab, nox], [w], lam, "TM")[0]


def ng_strip(w, h=0.22, lam=1.55, dl=0.005):
    n0 = neff_strip(w, h, lam)
    dn = (neff_strip(w, h, lam + dl) - neff_strip(w, h, lam - dl)) / (2 * dl)
    return n0 - lam * dn


# ---------------------------------------------------------------------------
# 2. Generic S-matrix netlist solver
# ---------------------------------------------------------------------------
def coupler(k, nw):
    """4-port lossless coupler, field coupling k. Ports: in1, in2, out1, out2."""
    t = np.sqrt(1 - k ** 2)
    S = np.zeros((nw, 4, 4), complex)
    S[:, 2, 0] = S[:, 0, 2] = t
    S[:, 3, 1] = S[:, 1, 3] = t
    S[:, 3, 0] = S[:, 0, 3] = 1j * k
    S[:, 2, 1] = S[:, 1, 2] = 1j * k
    return S, ["in1", "in2", "out1", "out2"]


def waveguide(L, wl, neff, loss_db_cm=2.0):
    """2-port waveguide. L and wl in um; neff is an array or a callable of wl."""
    n = neff(wl) if callable(neff) else neff
    e = np.exp(-2j * np.pi * n * L / wl) * 10 ** (-loss_db_cm * L * 1e-4 / 20)
    S = np.zeros((len(wl), 2, 2), complex)
    S[:, 0, 1] = S[:, 1, 0] = e
    return S, ["a", "b"]


def solve_circuit(insts, nets, wl):
    """insts: {name: (S[nw,p,p], port_names)}; nets: [("A.p", "B.q"), ...].
    Unconnected ports become external. Returns (S_ext[nw,E,E], ext_names)."""
    names, blocks, off = [], [], 0
    index = {}
    for iname, (S, ports) in insts.items():
        for j, pn in enumerate(ports):
            index[f"{iname}.{pn}"] = off + j
            names.append(f"{iname}.{pn}")
        blocks.append(S)
        off += len(ports)
    P, nw = off, len(wl)
    S = np.zeros((nw, P, P), complex)
    o = 0
    for B in blocks:
        p = B.shape[1]
        S[:, o:o + p, o:o + p] = B
        o += p
    C = np.zeros((P, P))
    for a, b in nets:
        C[index[a], index[b]] = C[index[b], index[a]] = 1
    internal = np.where(C.sum(1) > 0)[0]
    ext = np.where(C.sum(1) == 0)[0]
    Sii = S[:, internal][:, :, internal]
    Sie = S[:, internal][:, :, ext]
    Sei = S[:, ext][:, :, internal]
    See = S[:, ext][:, :, ext]
    Cii = C[np.ix_(internal, internal)]
    A = np.eye(len(internal)) - Cii @ Sii                 # (I - C S_ii)
    X = np.linalg.solve(A, Cii @ Sie)                     # a_i = (I - C S_ii)^-1 C S_ie a_e
    return See + Sei @ X, [names[i] for i in ext]


def port(Sext, names, out, inp):
    return Sext[:, names.index(out), names.index(inp)]


# ---------------------------------------------------------------------------
# Figure: EIM neff(w, lambda) and the polynomial compact model
# ---------------------------------------------------------------------------
def fig_neff_model():
    lam = np.linspace(1.50, 1.60, 21)
    widths = [0.40, 0.45, 0.50, 0.55, 0.60]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for w, c in zip(widths, [C5, C1, C3, C4, C2]):
        ne = np.array([neff_strip(w, lam=l) for l in lam])
        ax[0].plot(lam * 1e3, ne, "o", ms=4, color=c)
        cf = np.polyfit(lam - 1.55, ne, 2)
        ax[0].plot(lam * 1e3, np.polyval(cf, lam - 1.55), "-", color=c, label=f"w = {w*1e3:.0f} nm")
    ax[0].set(xlabel="wavelength (nm)", ylabel="$n_{eff}$", title="EIM data (dots) + quadratic model")
    ax[0].legend(fontsize=10)

    # fitting pitfalls: polynomial order vs spline, inside and outside the data range
    w_data = np.linspace(0.40, 0.60, 7)
    n_data = np.array([neff_strip(w) for w in w_data])
    w_fine = np.linspace(0.36, 0.66, 200)
    n_true = np.array([neff_strip(w) for w in w_fine])
    for deg, c in [(2, C1), (6, C2)]:
        cf = np.polyfit(w_data - 0.5, n_data, deg)
        ax[1].plot(w_fine * 1e3, (np.polyval(cf, w_fine - 0.5) - n_true) * 1e3, color=c,
                   label=f"polynomial, order {deg}")
    cs = CubicSpline(w_data, n_data, extrapolate=True)
    ax[1].plot(w_fine * 1e3, (cs(w_fine) - n_true) * 1e3, color=C3, label="cubic spline")
    ax[1].axvspan(400, 600, color="0.92", zorder=0)
    ax[1].text(500, 2.6, "data range", ha="center", color="0.4")
    ax[1].set(xlabel="width (nm)", ylabel=r"model error $\times 10^{3}$", ylim=(-3, 3),
              title="7 simulation points: interpolate, don't extrapolate")
    ax[1].legend(fontsize=10, loc="lower left")

    ng = [ng_strip(w) for w in w_fine[::10]]
    ne = n_true[::10]
    ax[2].plot(w_fine[::10] * 1e3, ne, "-o", ms=4, color=C1, label="$n_{eff}$")
    ax[2].plot(w_fine[::10] * 1e3, ng, "-o", ms=4, color=C2, label="$n_g = n_{eff}-\\lambda\\, dn_{eff}/d\\lambda$")
    ax[2].set(xlabel="width (nm)", ylabel="index", title="Effective and group index at 1550 nm")
    ax[2].legend(fontsize=10)
    fig.tight_layout()
    save(fig, "m6_neff_model.png")


# ---------------------------------------------------------------------------
# Figure: directional coupler from supermodes
# ---------------------------------------------------------------------------
def dc_kappa(gap, w=0.5, h=0.22, lam=1.55):
    nsi, nox = n_si(lam), n_sio2(lam)
    n_slab = slab_modes([nox, nsi, nox], [h], lam, "TE")[0]
    m = slab_modes([nox, n_slab, nox, n_slab, nox], [w, gap, w], lam, "TM")
    return np.pi * (m[0] - m[1]) / lam                         # 1/um


def fig_coupler():
    gaps = np.linspace(0.10, 0.40, 13)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for lam, c in [(1.50, C1), (1.55, C3), (1.60, C2)]:
        k = np.array([dc_kappa(g, lam=lam) for g in gaps])
        ax[0].semilogy(gaps * 1e3, np.pi / (2 * k), "o-", ms=4, color=c, label=f"λ = {lam*1e3:.0f} nm")
    ax[0].set(xlabel="gap (nm)", ylabel="cross-over length $L_c$ (µm)",
              title=r"$L_c=\pi/2\kappa=\lambda/2\Delta n$ (exponential in gap)")
    ax[0].legend(fontsize=10)
    L = np.linspace(0, 40, 300)
    for g, c in [(0.15, C2), (0.20, C3), (0.30, C1)]:
        k = dc_kappa(g)
        ax[1].plot(L, np.sin(k * L) ** 2, color=c, label=f"gap {g*1e3:.0f} nm")
    ax[1].set(xlabel="coupling length L (µm)", ylabel="cross power $\\sin^2(\\kappa L)$",
              title="Power exchange vs length")
    ax[1].legend(fontsize=10)
    lam = np.linspace(1.50, 1.60, 21)
    k0 = dc_kappa(0.2)
    Lc3 = np.pi / (4 * k0)
    for g, c in [(0.2, C3)]:
        k = np.array([dc_kappa(g, lam=l) for l in lam])
        ax[2].plot(lam * 1e3, np.sin(k * Lc3) ** 2, color=c, label="50:50 at 1550 nm")
        ax[2].plot(lam * 1e3, np.sin(k * 2 * Lc3) ** 2, color=C4, label="100% at 1550 nm")
    ax[2].set(xlabel="wavelength (nm)", ylabel="cross power", ylim=(0, 1.05),
              title="Wavelength dependence (gap 200 nm)")
    ax[2].legend(fontsize=10)
    fig.tight_layout()
    save(fig, "m6_coupler.png")


# ---------------------------------------------------------------------------
# Figure: ring resonator (all-pass) - amplitude, phase, group delay
# ---------------------------------------------------------------------------
def ring_allpass(phi, a, t):
    """All-pass ring: self-coupling t, round-trip amplitude a, round-trip phase phi."""
    return (t - a * np.exp(-1j * phi)) / (1 - a * t * np.exp(-1j * phi))


def fig_ring():
    phi = np.linspace(-0.3, 0.3, 2001)
    a = 0.98
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for t, c, lab in [(0.995, C1, "under-coupled (t > a)"), (0.98, C3, "critical (t = a)"),
                      (0.90, C2, "over-coupled (t < a)")]:
        H = ring_allpass(phi, a, t)
        ax[0].plot(phi, 10 * np.log10(np.abs(H) ** 2 + 1e-12), color=c, label=lab)
        ph = np.unwrap(np.angle(H))
        ax[1].plot(phi, ph - ph[0], color=c)
        if t != a:                                   # phase jumps by pi at critical coupling
            ax[2].plot(phi, -np.gradient(ph, phi), color=c)
    ax[0].set(xlabel="round-trip phase detuning φ (rad)", ylabel="|T|² (dB)", ylim=(-40, 1),
              title="Through-port transmission, a = 0.98")
    ax[0].legend(fontsize=10, loc="lower left")
    ax[1].set(xlabel="φ (rad)", ylabel="phase (rad)", title="Phase response (over-coupled: full 2π)")
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].set(xlabel="φ (rad)", ylabel="−dθ/dφ (round trips)",
              title="Group delay / round-trip time\n(over: slow light, under: negative)")
    fig.tight_layout()
    save(fig, "m6_ring.png")


# ---------------------------------------------------------------------------
# Figure: MZI built with the netlist solver vs. analytic formula
# ---------------------------------------------------------------------------
N0, NG0 = None, None


def neff_model():
    """Linear dispersion compact model n(lambda) = n0 - (ng - n0)(lambda - 1.55)/1.55."""
    global N0, NG0
    if N0 is None:
        N0, NG0 = neff_strip(0.5), ng_strip(0.5)
    return lambda wl: N0 - (NG0 - N0) * (wl - 1.55) / 1.55


def fig_mzi_netlist():
    wl = np.linspace(1.53, 1.57, 2000)
    nf = neff_model()
    k50 = np.sqrt(0.5)
    insts = {"dc1": coupler(k50, len(wl)), "dc2": coupler(k50, len(wl)),
             "arm1": waveguide(100.0, wl, nf), "arm2": waveguide(150.0, wl, nf)}
    nets = [("dc1.out1", "arm1.a"), ("arm1.b", "dc2.in1"),
            ("dc1.out2", "arm2.a"), ("arm2.b", "dc2.in2")]
    S, names = solve_circuit(insts, nets, wl)
    bar = np.abs(port(S, names, "dc2.out1", "dc1.in1")) ** 2
    cross = np.abs(port(S, names, "dc2.out2", "dc1.in1")) ** 2
    dphi = 2 * np.pi * nf(wl) * 50.0 / wl
    loss = 10 ** (-2.0 * 125e-4 / 10)
    analytic = loss * np.sin(dphi / 2) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    ax[0].plot(wl * 1e3, 10 * np.log10(bar), color=C1, label="bar (netlist solver)")
    ax[0].plot(wl * 1e3, 10 * np.log10(cross), color=C2, label="cross (netlist solver)")
    ax[0].plot(wl[::40] * 1e3, 10 * np.log10(analytic[::40]), "k.", ms=6, label="analytic $\\sin^2(\\Delta\\phi/2)$")
    ax[0].set(xlabel="wavelength (nm)", ylabel="transmission (dB)", ylim=(-35, 1),
              title=f"Unbalanced MZI, ΔL = 50 µm  →  FSR = λ²/(n_g ΔL) = {1.55**2/(NG0*50)*1e3:.1f} nm")
    ax[0].legend(fontsize=10, loc="lower right")
    ax[1].axis("off")
    y = 0.75
    ax[1].add_patch(FancyBboxPatch((0.05, 0.42), 0.25, 0.16, boxstyle="round,pad=0.02", fc="#dde8f3"))
    ax[1].add_patch(FancyBboxPatch((0.70, 0.42), 0.25, 0.16, boxstyle="round,pad=0.02", fc="#dde8f3"))
    ax[1].text(0.175, 0.5, "dc1", ha="center", va="center")
    ax[1].text(0.825, 0.5, "dc2", ha="center", va="center")
    ax[1].plot([0.32, 0.68], [0.56, 0.56], color=C1, lw=2)
    ax[1].plot([0.32, 0.35, 0.65, 0.68], [0.44, 0.2, 0.2, 0.44], color=C2, lw=2)
    ax[1].text(0.5, 0.65, "arm1: 100 µm", ha="center", color=C1)
    ax[1].text(0.5, 0.13, "arm2: 150 µm", ha="center", color=C2)
    ax[1].text(0.0, 0.56, "in1", ha="right", fontsize=10)
    ax[1].text(1.0, 0.56, "out1 (bar)", ha="left", fontsize=10)
    ax[1].text(1.0, 0.44, "out2 (cross)", ha="left", fontsize=10)
    ax[1].text(0.5, 0.9, "netlist = 4 instances + 4 nets", ha="center", fontsize=11)
    ax[1].set(xlim=(-0.2, 1.3), ylim=(0, 1))
    fig.tight_layout()
    save(fig, "m6_mzi_netlist.png")


# ---------------------------------------------------------------------------
# Figure: coupled-ring (higher-order) filters + 4-channel WDM demux
# ---------------------------------------------------------------------------
def coupled_rings(Kpow, wl, R=5.0, loss=2.0):
    """Series-coupled ring filter, power couplings Kpow[0..N]. Returns (thru, drop)."""
    nf = neff_model()
    N = len(Kpow) - 1
    insts, nets = {}, []
    half = np.pi * R
    for i, K in enumerate(Kpow):
        insts[f"c{i}"] = coupler(np.sqrt(K), len(wl))
    for i in range(1, N + 1):
        insts[f"r{i}a"] = waveguide(half, wl, nf, loss)
        insts[f"r{i}b"] = waveguide(half, wl, nf, loss)
        nets += [(f"c{i-1}.out2", f"r{i}a.a"), (f"r{i}a.b", f"c{i}.in1"),
                 (f"c{i}.out1", f"r{i}b.a"), (f"r{i}b.b", f"c{i-1}.in2")]
    S, names = solve_circuit(insts, nets, wl)
    return (np.abs(port(S, names, "c0.out1", "c0.in1")) ** 2,
            np.abs(port(S, names, f"c{N}.out2", "c0.in1")) ** 2)


def fig_filters():
    nf = neff_model()
    R = 5.0
    # find a resonance near 1550 to centre the plot
    m = np.round(nf(1.55) * 2 * np.pi * R / 1.55)
    lam_r = brentq(lambda l: nf(l) * 2 * np.pi * R / l - m, 1.53, 1.57)
    wl = lam_r + np.linspace(-2.0e-3, 2.0e-3, 3001)
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
    K1 = 0.12
    for Ks, c, lab in [([K1, K1], C1, "1st order"),
                       ([K1, 0.25 * K1 ** 2, K1], C3, "2nd order"),
                       ([K1, 0.125 * K1 ** 2, 0.125 * K1 ** 2, K1], C2, "3rd order")]:
        _, d = coupled_rings(Ks, wl, R)
        ax[0].plot((wl - lam_r) * 1e3, 10 * np.log10(d), color=c, label=lab)
    ax[0].set(xlabel="detuning (nm)", ylabel="drop port (dB)", ylim=(-50, 2),
              title="Series-coupled rings (R = 5 µm): flatter top, steeper skirts")
    ax[0].legend(fontsize=10)

    # 4-channel demux on a common bus, 200 GHz (~1.6 nm) spacing
    wl = lam_r + np.linspace(-5e-3, 5e-3, 4001)
    dlam = 1.6e-3
    insts, nets = {}, []
    radii = []
    for ch in range(4):
        target = lam_r + (ch - 1.5) * dlam
        Rc = m * target / (2 * np.pi * nf(target))
        radii.append(Rc)
        insts[f"b{ch}"] = coupler(np.sqrt(0.08), len(wl))
        insts[f"d{ch}"] = coupler(np.sqrt(0.08), len(wl))
        insts[f"ra{ch}"] = waveguide(np.pi * Rc, wl, nf)
        insts[f"rb{ch}"] = waveguide(np.pi * Rc, wl, nf)
        nets += [(f"b{ch}.out2", f"ra{ch}.a"), (f"ra{ch}.b", f"d{ch}.in1"),
                 (f"d{ch}.out1", f"rb{ch}.a"), (f"rb{ch}.b", f"b{ch}.in2")]
        if ch > 0:
            insts[f"bus{ch}"] = waveguide(20.0, wl, nf)
            nets += [(f"b{ch-1}.out1", f"bus{ch}.a"), (f"bus{ch}.b", f"b{ch}.in1")]
    S, names = solve_circuit(insts, nets, wl)
    for ch, c in zip(range(4), [C1, C3, C4, C2]):
        d = np.abs(port(S, names, f"d{ch}.out2", "b0.in1")) ** 2
        ax[1].plot(wl * 1e3, 10 * np.log10(d), color=c, label=f"drop {ch+1}: R = {radii[ch]*1e3:.1f} nm")
    th = np.abs(port(S, names, "b3.out1", "b0.in1")) ** 2
    ax[1].plot(wl * 1e3, 10 * np.log10(th), color="0.4", lw=1, label="through")
    ax[1].set(xlabel="wavelength (nm)", ylabel="transmission (dB)", ylim=(-40, 2),
              title="4-channel ring demux (200 GHz grid) from the netlist solver")
    ax[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
    fig.tight_layout()
    save(fig, "m6_filters.png")


# ---------------------------------------------------------------------------
# Figure: time-domain link (NRZ through a band-limited MZM) - eye diagrams
# ---------------------------------------------------------------------------
def eye_sim(f3db_ghz, br=25e9, nsym=600, sps=32, vpp=0.6, snr_db=24):
    bits = RNG.integers(0, 2, nsym)
    x = np.repeat(2 * bits - 1, sps).astype(float)
    dt = 1 / (br * sps)
    f = np.fft.fftfreq(len(x), dt)
    H = 1 / (1 + 1j * f / (f3db_ghz * 1e9))                 # 1st-order RC driver/electrode
    v = np.fft.ifft(np.fft.fft(x) * H).real * vpp / 2       # in units of V_pi
    P = 0.5 * (1 + np.sin(np.pi * v))                        # MZM at quadrature
    P += RNG.normal(0, P.std() * 10 ** (-snr_db / 20), len(P))
    Hrx = 1 / (1 + (1j * f / (0.75 * br)))                   # receiver bandwidth
    return np.fft.ifft(np.fft.fft(P) * Hrx).real, sps


def fig_eye():
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for a, f3, in zip(ax, [40, 18, 9]):
        y, sps = eye_sim(f3)
        span = 2 * sps
        t = np.arange(span) / sps
        for k in range(20, len(y) // sps - 3):
            a.plot(t, y[k * sps + sps // 2:k * sps + sps // 2 + span], color=C1, alpha=0.08, lw=1)
        a.set(xlabel="time (UI)", ylim=(-0.05, 1.05), title=f"25 Gb/s NRZ, modulator f$_{{3dB}}$ = {f3} GHz")
    ax[0].set_ylabel("received power (norm.)")
    fig.tight_layout()
    save(fig, "m6_eye.png")


# ---------------------------------------------------------------------------
# Figure: direct vs external modulation small-signal response
# ---------------------------------------------------------------------------
def fig_modulation():
    f = np.linspace(0.05, 60, 1200)
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
    for fr, c in [(4, C1), (8, C3), (12, C2)]:
        gam = 0.25e-9 * (fr * 1e9) ** 2 / (2 * np.pi) / 1e9 + 0.5   # K-factor damping gamma/2pi (GHz)
        H = fr ** 2 / (fr ** 2 - f ** 2 + 1j * f * gam)
        ax[0].plot(f, 20 * np.log10(np.abs(H)), color=c, label=f"DML, f$_r$ = {fr} GHz")
    ax[0].axhline(-3, color="0.5", ls=":")
    ax[0].set(xlabel="RF frequency (GHz)", ylabel="|H(f)| (dB)", ylim=(-20, 12), xlim=(0, 30),
              title="Directly modulated laser: relaxation-oscillation peak")
    ax[0].legend(fontsize=10)
    L = 0.3                                                       # cm
    for dn, a0, c, lab in [(0.0, 0.0, "k", "ideal match, no RF loss"),
                           (0.0, 1.2, C3, "RF loss 1.2 dB/(cm·√GHz)"),
                           (0.4, 1.2, C2, "+ index mismatch $n_{RF}-n_g$ = 0.4")]:
        al = a0 * np.sqrt(f) / 8.686                              # Np/cm (field)
        bm = 2 * np.pi * f * 1e9 * dn / (C0 * 100)                # rad/cm
        g = (al + 1j * bm) * L * 10
        H = np.where(np.abs(g) < 1e-9, 1, (1 - np.exp(-g)) / np.where(np.abs(g) < 1e-9, 1, g))
        ax[1].plot(f, 20 * np.log10(np.abs(H)), color=c, label=lab)
    ax[1].axhline(-3, color="0.5", ls=":")
    ax[1].set(xlabel="RF frequency (GHz)", ylabel="|H(f)| (dB, electrical)", ylim=(-12, 1),
              title="Travelling-wave MZM, 3 mm electrode")
    ax[1].legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    save(fig, "m6_modulation.png")


# ---------------------------------------------------------------------------
# Figure: analog link - gain, NF and SFDR vs photocurrent; two-tone test
# ---------------------------------------------------------------------------
KB, Q, T0 = 1.380649e-23, 1.602176634e-19, 290.0


def link_metrics(Idc, Vpi=3.0, RS=50.0, RL=50.0, rin_db=-155.0):
    """Quadrature-biased MZM link, unmatched detector into RL."""
    G = (np.pi * Idc / Vpi) ** 2 * RS * RL
    rin = 10 ** (rin_db / 10)
    Nout = G * KB * T0 + KB * T0 + 2 * Q * Idc * RL + rin * Idc ** 2 * RL   # W/Hz
    NF = 10 * np.log10(Nout / (G * KB * T0))
    OIP3 = 4 * Idc ** 2 * RL
    SFDR = 2 / 3 * 10 * np.log10(OIP3 / Nout)
    return 10 * np.log10(G), NF, SFDR, Nout


def fig_analog_link():
    Idc = np.logspace(-4, -1, 200)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for rin, c in [(-165, C1), (-155, C3), (-145, C2)]:
        G, NF, SFDR, _ = link_metrics(Idc, rin_db=rin)
        ax[1].semilogx(Idc * 1e3, NF, color=c, label=f"RIN {rin} dB/Hz")
        ax[2].semilogx(Idc * 1e3, SFDR, color=c)
    G, *_ = link_metrics(Idc)
    ax[0].semilogx(Idc * 1e3, G, color="k", label="V$_π$ = 3 V")
    G1, *_ = link_metrics(Idc, Vpi=1.0)
    ax[0].semilogx(Idc * 1e3, G1, color=C4, label="V$_π$ = 1 V")
    ax[0].axhline(0, color="0.6", ls=":")
    ax[0].set(xlabel="DC photocurrent $I_{dc}$ (mA)", ylabel="link gain (dB)", title=r"$G=(\pi I_{dc}/V_\pi)^2 R_S R_L$")
    ax[0].legend(fontsize=10)
    ax[1].set(xlabel="$I_{dc}$ (mA)", ylabel="noise figure (dB)", ylim=(0, 60), title="NF: thermal → shot → RIN limited")
    ax[1].legend(fontsize=10)
    ax[2].set(xlabel="$I_{dc}$ (mA)", ylabel="SFDR (dB·Hz$^{2/3}$)", title="Spurious-free dynamic range")
    fig.tight_layout()
    save(fig, "m6_analog_link.png")

    # two-tone test from the exact Bessel expansion
    Idc, Vpi, RS, RL = 10e-3, 3.0, 50.0, 50.0
    Pin_dbm = np.linspace(-40, 25, 300)
    Vamp = np.sqrt(2 * RS * 10 ** (Pin_dbm / 10) * 1e-3)
    x = np.pi * Vamp / Vpi
    i1 = 2 * Idc * jv(1, x) * jv(0, x)
    i3 = 2 * Idc * jv(2, x) * jv(1, x)
    P1 = 10 * np.log10(i1 ** 2 / 2 * RL / 1e-3)
    P3 = 10 * np.log10(i3 ** 2 / 2 * RL / 1e-3 + 1e-30)
    G, NF, SFDR, Nout = link_metrics(Idc, Vpi)
    nf_dbm = 10 * np.log10(Nout / 1e-3)
    lin1 = Pin_dbm + G
    oip3 = 10 * np.log10(4 * Idc ** 2 * RL / 1e-3)
    iip3 = oip3 - G
    lin3 = oip3 + 3 * (Pin_dbm - iip3)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(Pin_dbm, P1, color=C1, lw=2.5, label="fundamental (Bessel, exact)")
    ax.plot(Pin_dbm, P3, color=C2, lw=2.5, label="IM3 at 2f₁−f₂ (exact)")
    ax.plot(Pin_dbm, lin1, color=C1, ls="--", lw=1)
    ax.plot(Pin_dbm, lin3, color=C2, ls="--", lw=1)
    ax.axhline(nf_dbm, color="0.3", ls=":")
    ax.text(-20, nf_dbm + 3, "output noise floor (1 Hz bandwidth)", color="0.3")
    ax.plot(iip3, oip3, "ko")
    ax.annotate(f"IP3: IIP3 = {iip3:.1f} dBm", (iip3, oip3), xytext=(-15, oip3 + 12),
                arrowprops=dict(arrowstyle="->"))
    pin_sfdr = iip3 - (oip3 - nf_dbm) / 3
    ax.annotate("", xy=(pin_sfdr, nf_dbm), xytext=(pin_sfdr, nf_dbm + SFDR),
                arrowprops=dict(arrowstyle="<->", color=C3, lw=2))
    ax.text(pin_sfdr + 1, nf_dbm + SFDR / 2, f"SFDR = {SFDR:.0f} dB·Hz$^{{2/3}}$", color=C3)
    ax.set(xlabel="RF input power per tone (dBm)", ylabel="RF output power (dBm)", ylim=(nf_dbm - 15, oip3 + 25),
           title=f"Two-tone test: MZM link, I$_{{dc}}$ = 10 mA, V$_π$ = 3 V (G = {G:.1f} dB, NF = {NF:.1f} dB)")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    save(fig, "m6_two_tone.png")


# ---------------------------------------------------------------------------
# Figure: true-time delay vs phase-shifter beam steering (beam squint)
# ---------------------------------------------------------------------------
def fig_ttd():
    N, fc, th0 = 16, 10e9, np.deg2rad(30)
    d = C0 / fc / 2
    th = np.deg2rad(np.linspace(-90, 90, 2001))
    n = np.arange(N)[:, None]
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
    for f, c in [(6e9, C1), (10e9, C3), (14e9, C2)]:
        k = 2 * np.pi * f / C0
        ps = np.exp(1j * n * (k * d * np.sin(th) - 2 * np.pi * fc / C0 * d * np.sin(th0))).sum(0)
        tt = np.exp(1j * n * k * d * (np.sin(th) - np.sin(th0))).sum(0)
        ax[0].plot(np.rad2deg(th), 20 * np.log10(np.abs(ps) / N + 1e-6), color=c, label=f"{f/1e9:.0f} GHz")
        ax[1].plot(np.rad2deg(th), 20 * np.log10(np.abs(tt) / N + 1e-6), color=c, label=f"{f/1e9:.0f} GHz")
    for a, t in zip(ax, ["Phase shifters (set at 10 GHz): beam squints", "True-time delay: beam stays at 30°"]):
        a.axvline(30, color="0.5", ls=":")
        a.set(xlabel="angle θ (deg)", ylim=(-30, 1), xlim=(-10, 70), title=t)
        a.legend(fontsize=10)
    ax[0].set_ylabel("array factor (dB)")
    fig.tight_layout()
    save(fig, "m6_ttd.png")


# ---------------------------------------------------------------------------
# Figure: optical frequency combs
# ---------------------------------------------------------------------------
def fig_comb():
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    nn = np.arange(-12, 13)
    for a, beta in zip(ax[:2], [1.5, np.pi * 1.6]):
        p = jv(nn, beta) ** 2
        a.vlines(nn, -60, 10 * np.log10(p + 1e-12), color=C1, lw=3)
        a.set(xlabel="line index n  (f = f$_c$ + n f$_m$)", ylabel="line power (dB)", ylim=(-50, 2),
              title=f"Phase-modulation comb, β = {beta:.2f} rad: $|J_n(β)|^2$")
    # Kerr-soliton-like comb: sech^2 envelope, lines at f0 + n frep
    n = np.arange(-60, 61)
    env = 1 / np.cosh(n / 18) ** 2
    ax[2].vlines(n, -60, 10 * np.log10(env), color=C3, lw=1.5)
    ax[2].set(xlabel="mode number n", ylabel="line power (dB)", ylim=(-40, 2),
              title="Soliton microcomb: sech² envelope\n$f_n = f_0 + n f_{rep}$, f$_{rep}$ = 10 GHz–1 THz")
    fig.tight_layout()
    save(fig, "m6_comb.png")


# ---------------------------------------------------------------------------
# Figure: photonic RF processing - heterodyne generation and differentiation
# ---------------------------------------------------------------------------
def fig_rf_processing():
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    # heterodyne: two lasers 30 GHz apart with Lorentzian linewidths
    f = np.linspace(-60, 60, 4001)
    lor = lambda f0, dv, x=f: (dv / 2) ** 2 / ((x - f0) ** 2 + (dv / 2) ** 2)
    ax[0].plot(f, 10 * np.log10(lor(-15, 0.1) + lor(15, 0.1) + 1e-6), color=C1)
    ax[0].set(xlabel="optical frequency − 193.4 THz (GHz)", ylabel="power (dB)", ylim=(-50, 3),
              title="Two lasers, Δν = 30 GHz")
    fr = np.linspace(0, 60, 2001)
    ax[1].plot(fr, 10 * np.log10(lor(30, 0.2, fr) + 1e-6), color=C2, label="free-running: Δν$_{RF}$ = Δν$_1$+Δν$_2$")
    ax[1].plot(fr, 10 * np.log10(lor(30, 1e-3, fr) + 1e-6), color=C3, label="phase-locked / comb lines")
    ax[1].set(xlabel="RF frequency (GHz)", ylabel="RF power (dB)", ylim=(-50, 3),
              title=r"Photocurrent $\propto |E_1+E_2|^2 \Rightarrow \cos(2\pi\,\Delta\nu\, t)$")
    ax[1].legend(fontsize=9)
    # differentiator: critically coupled ring, pulse narrow compared to linewidth
    nt, dt = 2 ** 14, 0.5e-12
    t = (np.arange(nt) - nt / 2) * dt
    E = np.exp(-t ** 2 / (2 * (60e-12) ** 2))
    fq = np.fft.fftfreq(nt, dt)
    FSR, tc = 200e9, 0.90
    phi = 2 * np.pi * fq / FSR
    H = tc * (1 - np.exp(-1j * phi)) / (1 - tc ** 2 * np.exp(-1j * phi))
    Eo = np.fft.ifft(np.fft.fft(E) * H)
    dE = np.gradient(E, t)
    ax[2].plot(t * 1e12, E ** 2, color="0.5", label="input |E|²")
    ax[2].plot(t * 1e12, np.abs(Eo) ** 2 / np.abs(Eo).max() ** 2, color=C1, lw=2.5, label="ring output (norm.)")
    ax[2].plot(t * 1e12, dE ** 2 / (dE ** 2).max(), "k--", lw=1, label="|dE/dt|² (ideal)")
    ax[2].set(xlabel="time (ps)", ylabel="intensity (norm.)", xlim=(-250, 250),
              title="Ring at critical coupling = 1st-order differentiator")
    ax[2].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "m6_rf_processing.png")


# ---------------------------------------------------------------------------
# Figure: bend loss (Marcuse) for strip and rib waveguides
# ---------------------------------------------------------------------------
def marcuse_bend(R, n1, n2, w, lam):
    """Power loss coefficient (1/um) of a bent symmetric slab, TE (Marcuse)."""
    beta = 2 * np.pi / lam * slab_modes([n2, n1, n2], [w], lam, "TE")[0]
    k0 = 2 * np.pi / lam
    kap = np.sqrt(n1 ** 2 * k0 ** 2 - beta ** 2)
    gam = np.sqrt(beta ** 2 - n2 ** 2 * k0 ** 2)
    a = w / 2
    pre = gam ** 2 * kap ** 2 * np.exp(2 * gam * a) / (beta * (1 + gam * a) * (kap ** 2 + gam ** 2))
    return pre * np.exp(-2 * gam ** 3 / (3 * beta ** 2) * R)


def fig_bend():
    lam = 1.55
    nsi, nox = n_si(lam), n_sio2(lam)
    n220 = slab_modes([nox, nsi, nox], [0.22], lam, "TE")[0]
    n90 = slab_modes([nox, nsi, nox], [0.09], lam, "TE")[0]
    n150 = slab_modes([nox, nsi, nox], [0.15], lam, "TE")[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    for n2, R, c, lab in [(nox, np.linspace(0.8, 6, 200), C1, "strip 500×220 nm (oxide clad)"),
                          (n150, np.linspace(5, 120, 300), C3, "rib, 70 nm etch (150 nm slab)"),
                          (n90, np.linspace(2, 40, 300), C2, "rib, 130 nm etch (90 nm slab)")]:
        al = np.array([marcuse_bend(r, n220, n2, 0.5, lam) for r in R])
        db90 = 4.343 * al * np.pi * R / 2
        ax.loglog(R, db90, color=c, lw=2.5, label=lab)
    ax.axhline(0.01, color="0.5", ls=":")
    ax.text(1, 0.012, "0.01 dB / 90°", color="0.4")
    ax.set(xlabel="bend radius (µm)", ylabel="radiation loss (dB per 90°)", ylim=(1e-5, 10),
           title="Radiation loss only (Marcuse + EIM); add mode-mismatch & scattering in practice")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "m6_bend.png")


# ---------------------------------------------------------------------------
# Figure: thermal crosstalk (2D finite volume)
# ---------------------------------------------------------------------------
def heat2d(trench=False, h=0.25):
    xs = np.arange(-60, 60 + h / 2, h)
    t_sub, t_box, t_clad = 30.0, 2.0, 2.2
    ys = np.arange(h / 2, t_sub + t_box + t_clad, h)
    nx, ny = len(xs), len(ys)
    k = np.full((ny, nx), 130.0)
    k[ys > t_sub, :] = 1.4
    if trench:
        for x0 in (6.0, -10.0):
            mask_x = (xs >= x0) & (xs <= x0 + 4)
            k[np.ix_(ys > t_sub - 5, mask_x)] = 0.026
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * ny)

    def link(a, b, g):
        rows.extend([a, b]); cols.extend([b, a]); vals.extend([-g, -g])
        diag[a] += g; diag[b] += g

    kh = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    for j in range(ny):
        for i in range(nx - 1):
            link(idx[j, i], idx[j, i + 1], kh[j, i])
    kv = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
    for j in range(ny - 1):
        for i in range(nx):
            link(idx[j, i], idx[j + 1, i], kv[j, i])
    diag[idx[0, :]] += 2 * k[0, :]                              # T = 0 at heat sink
    A = sparse.coo_matrix((vals + list(diag), (rows + list(range(nx * ny)), cols + list(range(nx * ny)))),
                          shape=(nx * ny, nx * ny)).tocsr()
    q = np.zeros((ny, nx))
    heater = np.abs(xs) <= 1.5
    q[-1, heater] = 1.0 / heater.sum()                          # 1 W/m in total
    T = spsolve(A, q.ravel()).reshape(ny, nx)
    return xs, ys, T, t_sub


def fig_thermal():
    fig, ax = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    xs, ys, T, tsub = heat2d(False)
    jwg = np.argmin(np.abs(ys - (tsub + 2.11)))
    PL = 1e-3 / 100e-6                                            # 1 mW over 100 um -> 10 W/m
    im = ax[0].imshow(T * PL, origin="lower", aspect="auto", cmap="inferno",
                      extent=[xs[0], xs[-1], ys[0], ys[-1]])
    ax[0].set(ylim=(20, ys[-1]), xlim=(-30, 30), xlabel="x (µm)", ylabel="y (µm)",
              title="ΔT (K) for 1 mW in a 100 µm heater (no trench)")
    fig.colorbar(im, ax=ax[0])
    ax[0].axhline(tsub, color="w", ls=":", lw=1)
    ax[0].text(-29, tsub + 0.4, "BOX", color="w", fontsize=9)
    ax[0].text(-29, tsub - 1.4, "Si substrate", color="w", fontsize=9)
    dndT, lam, L = 1.86e-4, 1.55e-6, 100e-6
    for tr, c, lab in [(False, C1, "no trench"), (True, C2, "air trenches at x = 6–10 µm")]:
        xs, ys, T, tsub = heat2d(tr)
        dT = T[jwg] * PL
        ppi = (lam / (2 * L * dndT)) / dT[np.argmin(np.abs(xs))]
        ax[1].plot(xs, dT / dT[np.argmin(np.abs(xs))], color=c, lw=2.5, label=f"{lab}: P$_π$ ≈ {ppi:.0f} mW")
    ax[1].set(xlabel="distance from heater (µm)", ylabel="ΔT / ΔT(own waveguide)", xlim=(0, 50),
              title="Thermal crosstalk at waveguide level")
    ax[1].axhline(0.05, color="0.5", ls=":")
    ax[1].legend(fontsize=10)
    fig.tight_layout()
    save(fig, "m6_thermal.png")


# ---------------------------------------------------------------------------
# Figure: schematic PIC floorplan
# ---------------------------------------------------------------------------
def fig_layout():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.add_patch(Rectangle((0, 0), 5000, 2500, fc="#f7f7f7", ec="k", lw=1.5))
    for i in range(12):                                     # DC pads top
        ax.add_patch(Rectangle((1750 + i * 230, 2280), 100, 100, fc="#d4ac0d", ec="k"))
    for i in range(4):                                      # RF GSG pads
        for j, c in enumerate(["#888", "#d4ac0d", "#888"]):
            ax.add_patch(Rectangle((1500 + i * 700 + j * 110, 60), 80, 100, fc=c, ec="k"))
    for i in range(8):                                      # grating coupler array (127 um pitch)
        y = 400 + i * 127 * 2
        ax.add_patch(Rectangle((80, y - 25), 60, 50, fc="#5dade2", ec="k"))
        ax.plot([140, 500], [y, y], color=C1, lw=1.2)
    ax.text(250, 2330, "GC array (fibre array, 127/250 µm pitch)", fontsize=9, ha="left", va="center")
    # MZMs
    for i in range(4):
        y = 600 + i * 300
        ax.plot([500, 1200], [y, y], color=C1, lw=1.2)
        ax.add_patch(Rectangle((1300, y - 50), 2600, 100, fc="#fadbd8", ec=C2))
        ax.plot([1200, 4100], [y, y], color=C1, lw=1.2)
        ax.plot([1500 + i * 700 + 150, 1500 + i * 700 + 150], [160, y - 50], color="#d4ac0d", lw=2)
    ax.text(2600, 1600, "4 × travelling-wave MZM (RF from GSG pads, bottom)", ha="center", fontsize=10, color=C2)
    # ring bank + heaters
    for i in range(4):
        ax.add_patch(Circle((4300, 600 + i * 300), 40, fc="none", ec=C1, lw=2))
        ax.add_patch(Circle((4300, 600 + i * 300), 55, fc="none", ec="#d35400", lw=1, ls="--"))
        ax.plot([4300, 4300], [655 + i * 300, 2280], color="#d4ac0d", lw=0.8)
        ax.add_patch(Rectangle((4550, 570 + i * 300), 120, 60, fc="#a9cce3", ec="k"))
    ax.text(4250, 1720, "ring filters\n+ TiN heaters", ha="center", fontsize=10,
            bbox=dict(fc="#f7f7f7", ec="none"))
    ax.text(4610, 430, "Ge PDs", ha="center", fontsize=10)
    # test structures and marks
    ax.add_patch(Rectangle((600, 1800), 700, 330, fc="none", ec="0.4", ls="--"))
    ax.text(950, 1965, "test structures\n(loss cut-back, ring, GC ref.)", ha="center", va="center", fontsize=9)
    for (x, y) in [(150, 150), (4850, 150), (150, 2350 - 80), (4850, 2350 - 80)]:
        ax.plot([x - 40, x + 40], [y, y], "k", lw=2); ax.plot([x, x], [y - 40, y + 40], "k", lw=2)
    ax.annotate("thermal isolation\n≥ 50 µm / trenches", (4300, 900), xytext=(3300, 2000),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set(xlim=(-100, 5100), ylim=(-100, 2600), aspect="equal")
    ax.axis("off")
    ax.set_title("Example 5 × 2.5 mm transmitter/receiver floorplan (schematic, not to scale)")
    save(fig, "m6_layout.png")


# ---------------------------------------------------------------------------
# Figures: yield - Monte Carlo, corners, wafer map
# ---------------------------------------------------------------------------
def ring_lut():
    """LUT compact model n_eff(w, h) at 1550 nm (this is literally a PDK LUT)."""
    W = np.linspace(0.46, 0.54, 9)
    H = np.linspace(0.205, 0.235, 7)
    tab = np.array([[neff_strip(w, h) for h in H] for w in W])
    return RegularGridInterpolator((W, H), tab), ng_strip(0.5)


def fig_yield():
    lut, ng = ring_lut()
    n0 = lut([[0.5, 0.22]])[0]
    shift = lambda w, h: 1550 * (lut(np.c_[w, h]) - n0) / ng      # nm, resonance shift
    dw, dh = 1e-3, 1e-3
    sw = (shift(np.array([0.5 + dw]), np.array([0.22])) - shift(np.array([0.5 - dw]), np.array([0.22])))[0] / 2
    sh = (shift(np.array([0.5]), np.array([0.22 + dh])) - shift(np.array([0.5]), np.array([0.22 - dh])))[0] / 2
    N = 20000
    w = 0.5 + RNG.normal(0, 0.005, N)
    h = 0.22 + RNG.normal(0, 0.002, N)
    d = shift(w, h)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    ax[0].hist(d, bins=80, color=C1, alpha=0.8)
    ax[0].set(xlabel="resonance shift Δλ (nm)", ylabel="count",
              title=f"MC, σ_w = 5 nm, σ_h = 2 nm\n∂λ/∂w = {sw:.2f} nm/nm, ∂λ/∂h = {sh:.2f} nm/nm, σ = {d.std():.2f} nm")
    # corners
    ax[1].scatter((w[:3000] - 0.5) * 1e3, (h[:3000] - 0.22) * 1e3, s=3, color=C1, alpha=0.3)
    gw, gh = np.meshgrid(np.linspace(-20, 20, 81), np.linspace(-10, 10, 61))
    gd = shift(0.5 + gw.ravel() * 1e-3, 0.22 + gh.ravel() * 1e-3).reshape(gw.shape)
    cs = ax[1].contour(gw, gh, gd, levels=np.arange(-30, 31, 5), colors="0.4", linewidths=0.8)
    ax[1].clabel(cs, fmt="%d nm", fontsize=8)
    for cx, cy in [(-15, -6), (-15, 6), (15, -6), (15, 6)]:
        ax[1].plot(cx, cy, "s", color=C2, ms=9)
    ax[1].set(xlabel="Δw (nm)", ylabel="Δh (nm)", title="±3σ corners (red) vs MC cloud;\ncontours = Δλ (nm)")
    # yield vs heater tuning range: heaters only red-shift
    rng_nm = np.linspace(0, 12, 200)
    for offset, c in [(0.0, C2), (-3.0, C4), (-6.0, C3)]:
        need = -(d + offset)                     # resonance must be pushed red by 'need' (>0)
        y = [(np.mean((need >= 0) & (need <= r))) for r in rng_nm]
        ax[2].plot(rng_nm, np.array(y) * 100, color=c, lw=2.5,
                   label=f"design offset {offset:+.0f} nm")
    ax[2].set(xlabel="available red-shift tuning range (nm)", ylabel="yield (%)", ylim=(0, 102),
              title="Heaters only red-shift: centre the design on the blue side")
    ax[2].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "m6_yield.png")

    # wafer map: radial thickness profile + random
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    pitch, Rw = 10.0, 95.0
    xs = np.arange(-Rw, Rw + 1, pitch)
    X, Y = np.meshgrid(xs, xs)
    inside = np.hypot(X, Y) <= Rw - 7
    r = np.hypot(X, Y) / Rw
    hmap = 0.22 + (-0.003 + 0.006 * r ** 2) + RNG.normal(0, 0.0008, X.shape)
    wmap = 0.5 + 0.004 * X / Rw + RNG.normal(0, 0.002, X.shape)
    dmap = shift(wmap.ravel(), hmap.ravel()).reshape(X.shape)
    dmap[~inside] = np.nan
    im = ax[0].imshow(dmap, origin="lower", cmap="RdBu_r", vmin=-8, vmax=8,
                      extent=[xs[0] - 5, xs[-1] + 5, xs[0] - 5, xs[-1] + 5])
    ax[0].add_patch(Circle((0, 0), Rw + 5, fc="none", ec="k", lw=1.5))
    fig.colorbar(im, ax=ax[0], label="Δλ (nm)")
    ax[0].set(title="Ring resonance shift across a 200 mm wafer\n(radial thickness + width gradient + random)", xlabel="x (mm)", ylabel="y (mm)")
    ok = (dmap > -6) & (dmap < 2)
    col = np.where(ok, 1.0, 0.0)
    col[~inside] = np.nan
    ax[1].imshow(col, origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
                 extent=[xs[0] - 5, xs[-1] + 5, xs[0] - 5, xs[-1] + 5])
    ax[1].add_patch(Circle((0, 0), Rw + 5, fc="none", ec="k", lw=1.5))
    ax[1].set(title=f"Pass/fail for a −6…+2 nm window: yield = {100*np.nanmean(col):.0f} %", xlabel="x (mm)")
    fig.tight_layout()
    save(fig, "m6_wafer_map.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:                      # e.g. python module6_figures.py fig_ring fig_ttd
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    fig_neff_model()
    fig_coupler()
    fig_ring()
    fig_mzi_netlist()
    fig_filters()
    fig_eye()
    fig_modulation()
    fig_analog_link()
    fig_ttd()
    fig_comb()
    fig_rf_processing()
    fig_bend()
    fig_thermal()
    fig_layout()
    fig_yield()
