import numpy as np
import lime
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
from matplotlib import rc_context, pyplot as plt
import matplotlib.transforms as transforms

# Calculate the lookback time and the age of the universe at the given redshift
redshift = 4.299
lookback_time = cosmo.lookback_time(redshift)
age_of_universe = cosmo.age(redshift)
fig_folder = Path(f'/home/vital/Dropbox/Astrophysics/Tools/aspect')

# Print the results
print(f"Lookback time: {lookback_time:.2f}")
print(f"Age of the universe at redshift {redshift}: {age_of_universe:.2f}")

fig_cfg = {"figure.dpi" : 350,
           "figure.figsize" : (5, 4),
           "axes.labelsize": 25,
           "xtick.labelsize": 16,
           "ytick.labelsize": 20,
           "legend.fontsize":12,
            'font.family': 'Times New Roman',
           "text.usetex": True,
           }

output_folder=Path('/home/vital/Dropbox/Astrophysics/Seminars/2024_BootCamp')

spec_address = '/home/vital/PycharmProjects/ceers-data/data/spectra/CEERs_DR0.9/nirspecDDT/prism/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_x1d.fits'

spec = lime.Spectrum.from_file(spec_address, instrument='nirspec', redshift=redshift, crop_waves=(1.9, 2.03))
spec.unit_conversion('AA', 'FLAM')
# spec.plot.spectrum(show_err=True)
ax_label = lime.theme.ax_defaults(observation=spec, fig_type='default')
conf = lime.theme.fig_defaults(fig_cfg)
with rc_context(conf):
    fig, ax = plt.subplots()
    # ax.plot(spec.wave_rest, spec.flux, color='#ffe6cc')

    err_arr = spec.err_flux * 2
    err_arr[0], err_arr[1], err_arr[2], err_arr[3], err_arr[-2], err_arr[-1] = err_arr[0]*2, err_arr[1]*2.5, err_arr[2]*1.5, err_arr[3]*1.2, err_arr[-2]*1.75, err_arr[-1]*3
    err_min, err_max = spec.flux - err_arr, spec.flux + err_arr

    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, len(ax.get_yticks())))  # Match number of ticks
    ax2.set_ylabel('Scaled flux')
    ax.set_ylabel(ax_label['ylabel'])

    ax.step(spec.wave_rest, spec.flux, where='mid', color='#ffcccc',
            label=r'Box pixel flux $\mathbf{\left(F_{n, i}\right)}$')
    ax.fill_between(x=spec.wave_rest, y1=err_min, y2=err_max, step='mid', alpha=0.8, color='#ffe6cc', ec=None,
                    label=r'Box pixel uncertainty $\mathbf{\left(\delta F_{n, i}\right)}$')

    ax.set_xlabel('')
    ax.set_xticks([])  # Remove y-axis ticks
    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_folder/'line_uncertainty.png')

    # ax.step(spec.wave_rest, spec.flux, where='mid', color='#ffcccc',
    #         label=r'Box pixel flux $\mathbf{\left(F_{n, i}\right)}$')
    # ax.fill_between(x=spec.wave_rest, y1=err_min, y2=err_max, step='mid', alpha=0.8, color='#ffe6cc', ec=None,
    #                 label=r'Box pixel uncertainty $\mathbf{\left(\delta F_{n, i}\right)}$')
    #
    # ax.set(**ax_label)
    # ax.set_xlabel('')
    # ax.set_xticks([])  # Remove y-axis ticks
    # ax.set_xticklabels([])  #
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(fig_folder/'line_uncertainty.png')

    # scale = np.max(spec.flux+err_arr) - np.min(spec.flux-err_arr)
    # y_norm = (spec.flux - np.min(spec.flux-err_arr)) / scale
    # y_err_norm = err_arr / scale
    # err_min, err_max = y_norm - y_err_norm, y_norm + y_err_norm
    #
    # ax.step(spec.wave_rest, y_norm, where='mid', color='black', label=r'Box pixel flux $\mathbf{\left(F_{n, i}\right)}$')
    # ax.fill_between(x=spec.wave_rest, y1=err_min, y2=err_max, step='mid', alpha=1, color='#fffbd5', ec=None,
    #                 label=r'Box pixel uncertainty $\mathbf{\left(\delta F_{n, i}\right)}$')
    #
    #
    # ax_label['ylabel'] = 'Scaled flux'
    # ax.set(**ax_label)
    # ax.set_xlabel('')
    # ax.set_xticks([])  # Remove y-axis ticks
    # ax.set_xticklabels([])  #
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(fig_folder/'line_uncertainty_norm.png')

# box_params = dict(facecolor='none',
#                   edgecolor='#ffcccc',
#                   linestyle='--',
#                   linewidth=2)
# conf = lime.theme.fig_defaults(fig_cfg)
#
# with rc_context(conf):
#     fig = plt.figure(layout='constrained')
#     spec.plot.spectrum(rest_frame=True, in_fig=fig, fig_cfg=conf)
#     ax = spec.plot.ax
#
#     transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
#
#     y0, height = 0.1, 0.35
#     x0, width = 3300, 175
#     rect = plt.Rectangle((x0, y0), width=width, height=height, transform=transform, **box_params)
#     ax.add_patch(rect)
#
#     x1 = 3630
#     rect1 = plt.Rectangle((x1, y0), width=width, height=height, transform=transform, **box_params)
#     ax.add_patch(rect1)
#
#     # Arrow parameters: start above the top of the box
#     arrow_y_frac = y0 + height + 0.05  # Slightly above the box
#
#     ax.annotate('',
#                 xy=(x0 + 450, arrow_y_frac),
#                 xytext=(3380, arrow_y_frac),
#                 xycoords=transform,
#                 textcoords=transform,
#                 arrowprops=dict(arrowstyle='->', color='#ffcccc', mutation_scale=20, linewidth=2))
#
#     ax.set_ylabel('')
#     ax.set_yticks([])  # Remove y-axis ticks
#     ax.set_yticklabels([])  # Remove y-axis labels
#     ax.spines['top'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_xlim(3250, 5250)
#     plt.show()
#     # plt.savefig(fig_folder/'detector_motion.png')
