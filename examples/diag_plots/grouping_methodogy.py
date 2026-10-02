"""
Skeleton for Fig. 6 (Sect. 3.3, "Blind line grouping").

Layout
------
Row 1 : high resolving power spectrum                     (input)
Row 2 : low resolving power (PRISM-like) spectrum, sharex (Step 1: bands)
Row 3 : (a) bands pixel matrix W        (Step 2)
        (b) Gram matrix G = W W^T       (Step 3)
        (c) Rayleigh-like criterion     (Step 4) + LiMe output (Step 5)

Replace the SYNTHETIC SPECTRA block with the real DJA/CAPERS data and,
if preferred, the toy W with the true band-coverage matrix (the toy one
compresses the empty stretch between the two groups to keep it legible).
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import (Ellipse, Rectangle, ConnectionPatch,
                                FancyArrowPatch)

rng = np.random.default_rng(7)

PINK = '#e08a96'
GREEN = '#fffbd5ff'


# ------------------------------------------------------------------ lines
labels = [r'H$\beta$ 4861$\AA$', '[O III] 4959$\AA$', '[O III] 5007$\AA$',
          '[N II] 6548$\AA$', r'H$\alpha$ 6563$\AA$', '[N II] 6583$\AA$',
          '[S II] 6716$\AA$', '[S II] 6731$\AA$']
mu = np.array([4861., 4959., 5007., 6548., 6563., 6583., 6716., 6731.])
amp = np.array([1.00, 1.17, 3.50, 0.15, 2.86, 0.45, 0.55, 0.42])
n_lines = mu.size

# ------------------------------------------------- SYNTHETIC SPECTRA block
def gauss(w, m, s):
    return np.exp(-0.5 * ((w - m) / s) ** 2)

cont, sig_hi, sig_lo = 0.15, 3.0, 45.0

wave_hi = np.arange(4600., 6950., 1.0)
flux_hi = cont + rng.normal(0, 0.012, wave_hi.size)
for m, a in zip(mu, amp):
    flux_hi += a * gauss(wave_hi, m, sig_hi)

wave_lo = np.arange(4600., 6950., 22.0)              # coarse pixels
flux_lo = cont + rng.normal(0, 0.006, wave_lo.size)
for m, a in zip(mu, amp):                            # ~flux conserving
    flux_lo += a * (sig_hi / sig_lo) * gauss(wave_lo, m, sig_lo)

# ------------------------------------------------------------- toy W and G
# 8 lines x 18 schematic pixels; the two groups' pixel ranges overlap the
# way the real PRISM bands do (Hb/4959 share 1 px, 4959/5007 share 2, the
# [N II]+Halpha triplet shares most of its pixels, ...)
W = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Hbeta
              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # [O III]4959
              [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # [O III]5007
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],   # [N II]6548
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],   # Halpha
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],   # [N II]6583
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],   # [S II]6716
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])  # [S II]6731

G = W @ W.T

# Figure
fig = plt.figure(figsize=(12.0, 7.8))
outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.25], hspace=0.50,
                          left=0.10, right=0.985, top=0.965, bottom=0.105)
gs_top = outer[0].subgridspec(2, 1, hspace=0.12)
gs_bot = outer[1].subgridspec(1, 3, width_ratios=[1.5, 1.0, 1.0],
                              wspace=0.65)

ax_hi = fig.add_subplot(gs_top[0])
ax_lo = fig.add_subplot(gs_top[1], sharex=ax_hi)
ax_w = fig.add_subplot(gs_bot[0])
ax_g = fig.add_subplot(gs_bot[1])
ax_c = fig.add_subplot(gs_bot[2])

# Rows 1-2: spectra
ax_hi.plot(wave_hi, flux_hi, color='k', lw=0.9)
for m in mu: ax_hi.axvspan(m - 3.5 * sig_hi, m + 3.5 * sig_hi, color='#fffbd5ff', alpha=1, lw=0)
ax_hi.set_yticks([])
ax_hi.set_ylabel('High R\nspectrum', fontsize=12)

ax_lo.plot(wave_lo, flux_lo, color='k', lw=1.2, drawstyle='steps-mid')
for m in mu: ax_lo.axvspan(m - 2.5 * sig_lo, m + 2.5 * sig_lo, color='#fffbd5ff', alpha=1, lw=0)

ax_lo.set_ylim(top=0.48)
ax_lo.set_xticks([])
ax_lo.set_yticks([])
ax_lo.set_ylabel('Low R\nspectrum', fontsize=12)
ax_lo.set_xlabel('1) Determine the band widths for the expected transitions', fontsize=12)

# pink box around the Hbeta + [O III] group in the low-R panel
ax_lo.add_patch(Rectangle((4715, 0.03), 5155 - 4715, 0.94, transform=ax_lo.get_xaxis_transform(),
                          fill=False, ec=PINK, lw=2.0, zorder=6))

# Panel W matrix
cmap_w = ListedColormap(['white', '#fffbd5ff'])
ax_w.imshow(W, cmap=cmap_w, aspect='auto', vmin=0, vmax=1)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        value = str(W[i, j]) if j not in [7, 8] else '.'
        ax_w.text(j, i,value, ha='center', va='center', fontsize=8, color='black' if W[i, j] else '0.55')

ax_w.set_yticks(range(n_lines), labels=labels, fontsize=10)
ax_w.set_xticks([])
ax_w.set_xticks(np.arange(-.5, W.shape[1], 1), minor=True)
ax_w.set_yticks(np.arange(-.5, n_lines, 1), minor=True)
# ax_w.grid(which='minor', color='0.88', lw=0.5)
ax_w.tick_params(which='both', length=0)
# for s in ax_w.spines.values():
#     s.set_visible(False)
ax_w.set_xlabel('2) Get the binary bands-pixel matrix (W)', fontsize=12)

ax_w.set_title('Pixel array', fontsize=12)

ax_w.add_patch(Rectangle((-0.5, -0.5), 7, 3, fill=False, ec=PINK, lw=1.5, zorder=6))

# right-hand axis label describing the rows
ax_w2 = ax_w.twinx()
ax_w2.set_ylim(ax_w.get_ylim())
ax_w2.set_yticks([])
ax_w2.set_ylabel('Line list', fontsize=12, labelpad=6)
for spine in ax_w2.spines.values():
    spine.set_visible(False)


# --------------------------------------------------- panel (b): Gram matrix
ax_g.set_xlim(-0.5, n_lines - 0.5)
ax_g.set_ylim(n_lines - 0.5, -0.5)
for i in range(n_lines):
    for j in range(n_lines):
        ax_g.text(j, i, str(G[i, j]), ha='center', va='center',
                  fontsize=9, color='k' if j >= i else '0.65')


def diag_ellipse(ax, point_a, point_b, pad_along=0.55, thickness=0.62,
                 **kwargs):
    """Empty ellipse whose major axis runs through two (row, col) points."""
    yA, xA = point_a  # unpack as (row, col) per the user's convention
    yB, xB = point_b
    cx, cy = (xA + xB) / 2, (yA + yB) / 2
    dx, dy = xB - xA, yB - yA
    length = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))
    ax.add_patch(Ellipse((cx, cy), length + pad_along, thickness, angle=angle, fill=False, **kwargs))


diag_ellipse(ax_g, (0, 1), (1, 2), ec=PINK, lw=1.5)
diag_ellipse(ax_g, (3, 4), (6, 7), ec=PINK, lw=1.5)
ax_g.set_xticks(range(n_lines), labels=labels, rotation=60, ha='left',
                rotation_mode='anchor', fontsize=9)
ax_g.xaxis.set_ticks_position('top')
ax_g.set_yticks(range(n_lines), labels=labels, fontsize=10)
ax_g.tick_params(length=0)
# for s in ax_g.spines.values():
#     s.set_visible(False)
ax_g.set_xlabel('3) Compute Gram matrix\n$G = W\\,W^{T}$', fontsize=12, labelpad=6)

# ---------------------------------------------- panel (c): Rayleigh proxy
x = np.linspace(-4.2, 4.2, 500)
ax_c.set_ylim(-0.05, 3.0)
ax_c.set_xticks([])
ax_c.set_yticks([])
# for s in ax_c.spines.values():
#     s.set_visible(False)

# aligned two-line system (mimics a LaTeX align env: right-align both
# fractions on a shared x so the "=>" arrows line up), with a tall "{"
# drawn to its left to visually group the two conditions
eq_x = 0.50  # shared anchor: right edge of the fraction column
ax_c.text(eq_x, 0.90,r'$\dfrac{\Delta\mu}{\sigma_{rms}}<2\Rightarrow$', transform=ax_c.transAxes, ha='right', va='center', fontsize=9.5)
ax_c.text(eq_x, 0.78,r'$\dfrac{\Delta\mu}{\sigma_{rms}}>2\Rightarrow$', transform=ax_c.transAxes, ha='right', va='center', fontsize=9.5)
ax_c.text(eq_x + 0.03, 0.90, 'unresolved', transform=ax_c.transAxes, ha='left', va='center', fontsize=9.5)
ax_c.text(eq_x + 0.03, 0.78, 'resolved', transform=ax_c.transAxes, ha='left', va='center', fontsize=9.5)

for dmu, lab, c in [(1.00, '1.0', '#fff7aeff'),
                    (2.00, '2.0', '#ffe6ccff'),
                    (2.35, '2.35', '#e08a96')]:
    ax_c.plot(x, gauss(x, -dmu / 2, 1) + gauss(x, dmu / 2, 1), color=c, lw=2, label=lab)

ax_c.legend(title=r'$\Delta\mu/\sigma_{\rm rms}$', loc='upper center', bbox_to_anchor=(0.5, 0.73), fontsize=7, title_fontsize=7.5,
            frameon=False, ncol=3, columnspacing=0.9, handlelength=1.1, borderaxespad=0.0, labelspacing=0.25)

ax_c.set_title('Merged | blended diagnostic', fontsize=11)
ax_c.set_xlabel('4) Determine the resolved\nstatus of consecutive lines', fontsize=12, labelpad=6)

# ------------------------------------- pink connector + bottom flow arrow


box_x_data = 4750  # center of the pink box, ax_lo data x
box_y_frac = 0.03  # bottom edge, ax_lo axes-fraction y

fig.canvas.draw()  # finalize layout so transforms below are accurate
xA_disp, yA_disp = ax_lo.get_xaxis_transform().transform((box_x_data, box_y_frac))
xA_fig, yA_fig = fig.transFigure.inverted().transform((xA_disp, yA_disp))

pw = ax_w.get_position()
yB_fig = pw.y1 + 0.00  # how far above panel (a) the line ends; tweak to taste

fig.add_artist(ConnectionPatch(xyA=(xA_fig, yA_fig), coordsA=fig.transFigure, xyB=(xA_fig, yB_fig), coordsB=fig.transFigure,
                color=PINK, lw=1.8, zorder=1))

pw, pc = ax_w.get_position(), ax_c.get_position()
fig.add_artist(FancyArrowPatch((pw.x0, 0.020), (pc.x1 + 0.005, 0.020), transform=fig.transFigure,
    arrowstyle='-|>', mutation_scale=16, lw=2.5, color=PINK, zorder=0))

fig.savefig('line_grouping.png', dpi=400)
print('done')