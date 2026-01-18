import numpy as np
import lime
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
from matplotlib import rc_context, pyplot as plt
import matplotlib.transforms as transforms

lime.theme.set_style('dark')

# Calculate the lookback time and the age of the universe at the given redshift
redshift = 4.299
lookback_time = cosmo.lookback_time(redshift)
age_of_universe = cosmo.age(redshift)
fig_folder = Path(f'/home/vital/Dropbox/Astrophysics/Tools/aspect')

# Print the results
print(f"Lookback time: {lookback_time:.2f}")
print(f"Age of the universe at redshift {redshift}: {age_of_universe:.2f}")

fig_cfg = {"figure.dpi" : 350,
           "figure.figsize" : (8, 3),
           "axes.labelsize": 16, "xtick.labelsize": 14,
            'font.family': 'Times New Roman',
           "text.usetex": True,
           }

output_folder=Path('/home/vital/Dropbox/Astrophysics/Seminars/2024_BootCamp')

spec_address = '/home/vital/Dropbox/Astrophysics/Data/CEERs/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_x1d.fits'
spec = lime.Spectrum.from_file(spec_address, instrument='nirspec', redshift=redshift, crop_waves=(0.75, 5.2))
spec.unit_conversion('AA', 'FLAM')

box_params = dict(facecolor='none',
                  edgecolor='#ffcccc',
                  linestyle='--',
                  linewidth=2)
conf = lime.theme.fig_defaults(fig_cfg)

with rc_context(conf):
    fig = plt.figure(layout='constrained')
    spec.plot.spectrum(rest_frame=True, in_fig=fig, fig_cfg=conf)
    ax = spec.plot.ax

    transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    y0, height = 0.1, 0.35
    x0, width = 3300, 175
    rect = plt.Rectangle((x0, y0), width=width, height=height, transform=transform, **box_params)
    ax.add_patch(rect)

    x1 = 3630
    rect1 = plt.Rectangle((x1, y0), width=width, height=height, transform=transform, **box_params)
    ax.add_patch(rect1)

    # Arrow parameters: start above the top of the box
    arrow_y_frac = y0 + height + 0.05  # Slightly above the box

    ax.annotate('',
                xy=(x0 + 450, arrow_y_frac),
                xytext=(3380, arrow_y_frac),
                xycoords=transform,
                textcoords=transform,
                arrowprops=dict(arrowstyle='->', color='#ffcccc', mutation_scale=20, linewidth=2))

    ax.set_ylabel('')
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(3250, 5250)
    # plt.show()
    plt.savefig(fig_folder/'detector_motion_dark.png')
