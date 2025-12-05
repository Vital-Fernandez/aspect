import lime
import numpy as np
import matplotlib.pyplot as plt
from lime.fitting.lines import c_KMpS
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import BlendedGenericTransform, blended_transform_factory

# Helper function to transfer artists

def transfer_artists(source_fig, dest_ax, idx_src_ax=None, close_src_fig=True):
    source_ax = source_fig.axes[0 if idx_src_ax is None else idx_src_ax]

    for artist_list_name in ['lines', 'collections', 'patches', 'images', 'texts']:
        artists_to_move = getattr(source_ax, artist_list_name)[:]

        for artist in artists_to_move:
            tr = artist.get_transform()

            artist.remove()
            artist.axes = dest_ax

            # Case 1: pure data transform
            if tr == source_ax.transData:
                artist.set_transform(dest_ax.transData)

            # Case 2: axvspan / axhspan-style blended transform
            elif isinstance(tr, BlendedGenericTransform):
                # rebuild same kind of blended transform on dest_ax
                artist.set_transform(
                    blended_transform_factory(dest_ax.transData, dest_ax.transAxes)
                )

            dest_ax.add_artist(artist)

    # Transfer axis properties
    dest_ax.set_xlim(source_ax.get_xlim())
    dest_ax.set_ylim(source_ax.get_ylim())
    dest_ax.set_xlabel(source_ax.get_xlabel())
    dest_ax.set_ylabel(source_ax.get_ylabel())
    dest_ax.set_title(source_ax.get_title())

    handles, labels = source_ax.get_legend_handles_labels()
    if handles:
        dest_ax.legend(handles, labels, loc='best')

    if close_src_fig:
        plt.close(source_fig)

    return


def artificial_spec(wave_min = 4800.0, wave_max=7000, R=5000, sigma=5, line_list=None, line_amps=None):
    # --- Settings ---

    # wave_max = 7000.0
    # R = 5000.0

    # # Emission line list and amplitudes
    # line_waves = np.array([4861.0, 4959.0, 5007.0, 6563.0, 6717.0, 6731.0])
    # line_amps = np.array([3.0, 4.0, 12.0, 6.0, 1.0, 1.0])
    line_waves = lime.label_decomposition(line_list, params_list=['wavelength'], scalar_output=True)[0]

    # --- Build wavelength grid with variable spacing ---
    print('step', wave_min/R)
    wave = np.arange(wave_min, wave_max, step=wave_min/R)

    # --- Flat continuum ---
    flux = np.ones_like(wave) + np.random.normal(0, 0.05, size=wave.size)

    sigma_pixels = sigma/c_KMpS * R
    sigma_pixels = sigma
    print('sigma_pixels', sigma_pixels)

    # --- Add Gaussian emission lines ---
    if (line_waves is not None) and (line_amps is not None):
        for lam0, amp in zip(line_waves, line_amps):
            mu = wave[np.searchsorted(wave, lam0)]
            line_profile = amp * np.exp(-0.5 * ((wave - mu) / sigma_pixels)**2)
            flux += line_profile

    # wave, flux = artificial_spec(R=R, line_waves=line_waves, line_amps=line_amps)
    spec = lime.Spectrum(wave, flux, redshift=0)

    return spec


# Generate the spectrum
R = 100
line_list = ['H1_4861A', 'O3_4959A', 'O3_5007A', 'H1_6563A', 'S2_6716A', 'S2_6731A']
line_amps = np.array([3.0, 4.0, 12.0, 6.0, 0.5, 0.5])

spec_low = artificial_spec(R=80, line_list=line_list, line_amps=line_amps)
spec_high = artificial_spec(R=1000, line_list=line_list, line_amps=line_amps)

# Generate the bands
cfg = {'Ar4_4711A_b':   'Ar4_4711A+He1_4713A',
       'Ar4_4713A_m':   'Ar4_4711A+He1_4713A',
       'H1-O3_4861A_b': 'H1_4861A+O3_5007A_m',
       'H1_4861A_b':    'H1_4861A+O3_4959A+O3_5007A',
       'H1_4861A_m':    'H1_4861A+O3_4959A+O3_5007A',
       'O3_5007A_b':    'O3_4959A+O3_5007A',
       'O3_5007A_m':    'O3_4959A+O3_5007A',
       'H1-S2_6563A_b': 'H1_6563A+S2_6716A_m',
       'H1_6563A_m':    'H1_6563A+S2_6716A+S2_6731A',
       'S2_6716A_m':    'S2_6716A+S2_6731A',
       'S2_6716A_b':    'S2_6716A+S2_6731A'}

# Plot the bands
# spec.plot.spectrum(bands=bands,  ax_cfg={'title': f'Resolving power {R} ({wave.size} pixels)'})
# spec.plot.spectrum(bands=bands_grouped,  ax_cfg={'title': f'Resolving power {R} ({wave.size} pixels)'})

# spec_low.plot.spectrum(bands=bands_grouped, in_fig=None)
# spec_high.plot.spectrum(bands=bands, in_fig=None)

# inset = inset_axes(spec_low.plot.ax, width="50%", height="30%", loc="upper center", borderpad=0.3)
# transfer_artists(spec_high.plot.fig, inset)
#
# # --- Step 5: apply the inset locator ---
# inset.axis('off')

lime.theme.plt['spectrum_width'] = 0.5
# spec, label, velocity = spec_low, 'low resolving power', 300
spec, label, velocity = spec_high, 'high resolving power', 170
bands = spec.retrieve.lines_frame(band_vsigma=velocity, line_list=line_list, fit_cfg=cfg, default_cfg_prefix=None, automatic_grouping=True)
spec.plot.spectrum(bands=bands, in_fig=None, fig_cfg={"figure.dpi" : 600, "figure.figsize" : (8, 2)})
print(spec.wave.size)

for t in spec.plot.ax.texts: t.remove()

spec.plot.ax.update({'xlabel': label, 'ylabel': ''})
# spec.plot.ax.update({'ylabel': ''})
spec.plot.ax.spines['top'].set_visible(False)
spec.plot.ax.spines['bottom'].set_visible(False)
spec.plot.ax.spines['right'].set_visible(False)
spec.plot.ax.spines['left'].set_visible(False)
spec.plot.ax.tick_params(axis='both', which='both', bottom=False, top=False,  left=False,  right=False,  labelbottom=False,
                         labelleft=False)
# spec_low.plot.show()
plt.savefig('bands_example_large_res.png', bbox_inches='tight')

