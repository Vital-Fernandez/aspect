import numpy as np
from matplotlib import pyplot as plt, rc_context
import lime


# Parameters
sigma = 1.0            # Standard deviation of the Gaussians
A = 1.0                # Amplitude
x = np.linspace(-6, 6, 800)

# Convert sigma to FWHM
fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
print(2 * np.sqrt(2 * np.log(2)) * sigma * 3)

# Different separations to show
sep_arr = np.array([1*sigma, 2*sigma, 2 * np.sqrt(2 * np.log(2)) * sigma])
fwhm_arr = 2 * np.sqrt(2 * np.log(2)) * sigma

with rc_context(lime.theme.fig_defaults({"figure.dpi": 600, "figure.figsize": [5, 3], 'legend.fontsize': 9})):

    fig, ax = plt.subplots()

    for d in sep_arr:
        g1 = A * np.exp(-(x - d/2)**2 / (2 * sigma**2))
        g2 = A * np.exp(-(x + d/2)**2 / (2 * sigma**2))
        formula1, formula2 = r'$\Delta \mu$', r'$\sigma_{rms}$'
        ax.plot(x, g1 + g2, label=f" Peak separation ({formula1}) = {d:.2f}{formula2} = {d/fwhm_arr:.1f}FWHM)")

    ax.set_xlabel("x")
    ax.set_ylim(-0.25, 2.00)
    ax.axis('off')
    ax.legend(loc='lower center', ncol=1, framealpha=1, bbox_to_anchor=(0.5, 0.15))
    plt.savefig('gaussians_separations.png', bbox_inches='tight')
