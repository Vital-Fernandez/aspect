import numpy as np
from matplotlib import pyplot as plt, gridspec, colors, rc_context
from .io import Aspect_Error, cfg
from lime.plotting.plots import theme
from .tools import detection_function, stratify_sample



def decision_matrix_plot(matrix_arr, output_address=None, categories=None, exclude_diagonal=True, show_categories=False):

    # Try to find from database if None provided
    matrix_name = None
    if isinstance(matrix_arr, str):
        if matrix_arr in cfg['decision_matrices']:
            matrix_name = str(matrix_arr)
            matrix_arr = np.array(cfg['decision_matrices'][matrix_arr])
        else:
            raise Aspect_Error(f'Decision matrix "{matrix_arr}" not found in configuration file please')
        categories = list(cfg['shape_number'].keys())

    # Number of categories
    n_categories = matrix_arr.shape[0]

    # Set the diagonal to a distinct value (e.g., -1) to differentiate it
    if exclude_diagonal:
        np.fill_diagonal(matrix_arr, -1)

    # Figure format
    decision_colors = cfg['decision_matrices']['decision_colors']
    axes_labels = None if matrix_name is None else cfg['decision_matrices'][f'{matrix_name}_labels']

    # Start the figure
    with rc_context(cfg['plots_format']['dm']):

        # Define colors for values
        cmap = colors.ListedColormap(['white', decision_colors[0], decision_colors[1]])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Adjusting the plot by adding gridlines to the subplots
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(n_categories, 2, width_ratios=[n_categories, 1], wspace=0.05)

        # Plot the decision matrix on the left (column 0 of the GridSpec)
        ax_matrix = fig.add_subplot(gs[:, 0])
        ax_matrix.matshow(matrix_arr, cmap=cmap, norm=norm)

        # Customize matrix ticks and labels
        ax_matrix.set_xticks(range(n_categories))
        ax_matrix.set_yticks(range(n_categories))

        ax_matrix.set_xticklabels(categories, rotation=90)
        ax_matrix.set_yticklabels(categories)

        # # Move x-axis labels to the bottom
        # ax_matrix.xaxis.set_ticks_position('bottom')
        # ax_matrix.xaxis.set_label_position('bottom')

        # Add black gridlines to separate each square
        ax_matrix.set_xticks(np.arange(-.5, n_categories, 1), minor=True)
        ax_matrix.set_yticks(np.arange(-.5, n_categories, 1), minor=True)
        ax_matrix.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax_matrix.tick_params(which="minor", size=0)

        if axes_labels is not None:
            ax_matrix.set_ylabel(axes_labels[0], color=decision_colors[0])
            ax_matrix.set_xlabel(axes_labels[1], color=decision_colors[1])

        # Add individual plots on the right side (column 1 of the GridSpec) for each category
        if show_categories:

            for i, category in enumerate(categories):

                ax_plot = fig.add_subplot(gs[i, 1])

                # Example placeholder data for each category plot
                x = np.linspace(0, 10, 100)
                y = np.sin(x + i)  # Example sine wave that varies by row
                ax_plot.step(x, y, color=cfg['colors'][category])

                # Add gridlines to the subplots
                ax_plot.grid(True, which='both', linestyle='--', linewidth=0.5)

                # Hide y-axis ticks for these small plots for a cleaner look
                ax_plot.yaxis.set_ticks([])
                ax_plot.xaxis.set_ticks([])

        # Output the result
        if output_address is None:
            plt.show()
        else:
            fig.savefig(output_address, bbox_inches='tight')

    return


def scatter_plot(fig, ax, x_arr, y_arr, labels_arr, feature_list, color_dict, alpha=0.5, idx_target=None,
                 detection_range=None, ratio_color=None):


    if ratio_color is not None:
        scatter = ax.scatter(x_arr, y_arr, c=ratio_color, cmap='viridis', alpha=alpha, edgecolor='none',
                             norm=colors.LogNorm(vmin=ratio_color.min(), vmax=ratio_color.max()))
        cbar = fig.colorbar(scatter, ax=ax)  # Ensure the colorbar is linked to the figure
        cbar.set_label('Max - Min')

        # c = ax.scatter(x, y, c=magnitude, cmap='viridis',
        #                norm=colors.LogNorm(vmin=magnitude.min(), vmax=magnitude.max()), s=50, edgecolor='k')


    else:

        for i, feature in enumerate(feature_list):
            idcs_class = labels_arr == feature
            x_feature = x_arr[idcs_class]
            y_feature = y_arr[idcs_class]
            label = f'{feature} ({y_feature.size})'
            color_points = color_dict[feature]# if ratio_color is None else ratio_color[idcs_class]
            scatter = ax.scatter(x_feature, y_feature, label=label, c=color_points, alpha=alpha, edgecolor='none')

            # if ratio_color is None:
            #     scatter = ax.scatter(x_feature, y_feature, label=label, c=color_points, alpha=alpha, edgecolor='none')
            # else:
            #     scatter = ax.scatter(x_feature, y_feature, label=label, c=color_points, cmap='viridis', alpha=alpha,
            #                          edgecolor='none')
            #
            #     # Color bar just one
            #     if i == 0:
            #         cbar = fig.colorbar(scatter, ax=ax)  # Ensure the colorbar is linked to the figure
            #         cbar.set_label('Max - Min')

            # if ratio_color is not None:
            #     fig.colorbar(scatter, ax=ax)

    if idx_target is not None:
        ax.scatter(x_arr[idx_target], y_arr[idx_target], marker='x', label='selection', color='black')

    if detection_range is not None:
        ax.plot(detection_range, detection_function(detection_range))

    return

def parse_fig_cfg(fig_cfg=None, ax_diag=None, ax_line=None, dtype=None):

    # Input configuration updates default
    fig_cfg = fig_cfg if fig_cfg is not None else {'axes.labelsize': 10, 'axes.titlesize': 10,
                                                'figure.figsize': (12, 6), 'hatch.linewidth': 0.3, 'legend.fontsize': 8}
    fig_cfg = theme.fig_defaults(fig_cfg)

    if dtype == 'classifier':
        ax_diag = {} if ax_diag is None else ax_diag
        ax_diag = {'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
                   'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)',
                   **ax_diag}

        ax_line = {} if ax_line is None else ax_line
        ax_line = {'xlabel': 'Feature Number', 'ylabel': 'value', **ax_line}

    if dtype == 'doublet':
        ax_diag = {} if ax_diag is None else ax_diag
        ax_diag = {'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
                   'ylabel': r'$S_{pixels}$ (pixels)',
                   **ax_diag}

        ax_line = {} if ax_line is None else ax_line
        ax_line = {'xlabel': 'Feature Number', 'ylabel': 'value', **ax_line}

    return {'fig': fig_cfg, 'ax1': ax_diag, 'ax2': ax_line}


def ax_wording(ax, ax_cfg=None, legend_cfg=None, yscale=None):

    ax.update(ax_cfg)

    if legend_cfg is not None:
        ax.legend(**legend_cfg)

    if yscale is not None:
        ax.set_yscale(yscale)

    return


def plot_steps(spec, y_norm, idx, counts, model_mgr, out_type, seg_pred, old_pred):
    print(idx)
    x_arr = spec.wave_rest.data[spec.wave_rest.mask]
    x_sect = x_arr[idx:idx+y_norm.shape[0]]
    print(f'Idx "{idx}"; counts: {counts}; Output: {model_mgr.medium.number_feature_dict[out_type]} ({out_type})')

    colors_old = [cfg['colors'][model_mgr.medium.number_feature_dict[val]] for val in old_pred]
    colors_new = [cfg['colors'][model_mgr.medium.number_feature_dict[val]] for val in seg_pred]

    fig, ax = plt.subplots()
    color_detection = cfg['colors'][model_mgr.medium.number_feature_dict[out_type]]
    ax.step(x_sect, y_norm[:,0], where='mid', color=color_detection, label='Out detection')
    ax.scatter(x_sect, np.zeros(x_sect.size), color=colors_old, label='Old prediction')
    ax.scatter(x_sect, np.ones(x_sect.size), color=colors_new, label='New prediction')
    ax.set_xlabel(r'Wavelength $(\AA)$')

    ax_secondary = ax.twinx()  # Creates a twin y-axis on the right
    ax_secondary.set_ylim(ax.get_ylim())  # Match the primary y-axis limits
    ax_secondary.set_yticks([0, 0.5, 1])  # Custom tick positions
    ax_secondary.set_yticklabels(['Previous\nClassification', 'Present\nClassification', 'Output\nClassification'])

    plt.tight_layout()
    plt.show()

    return

class CheckSample:

    def __init__(self, in_data_arr, in_pred_arr, idx_features, fig_cfg=None, ax_diag=None, ax_line=None, base=10000,
                 sample_size=None, categories=None, column_labels='shape_class', dtype='classifier', ):

        # Stratify the selection aiming for same number of points
        data_arr, pred_arr = stratify_sample(in_data_arr, in_pred_arr, sample_size, categories, randomize=True)

        self.y_base = base
        self.x_coords = data_arr[:, 1]
        self.y_coords = data_arr[:, 0]
        self.id_arr = pred_arr
        self.classes = np.sort(np.unique(self.id_arr))
        self.data_df = data_arr[:, idx_features:]
        self.wave_range = np.arange(self.data_df.shape[1])
        self.dtype = dtype
        self.y_scale = 'log' if self.dtype == 'classifier' else 'linear'
        self.ratio_color = None
        # self.ratio_color = (data_arr[:, 3] - data_arr[:, 2])/10000#  - data_arr[:, 2])
        # self.ratio_color = (data_arr[:, 3])/10000#  - data_arr[:, 2])
        # self.ratio_color = (data_arr[:, 3]-data_arr[:, 2])/10000#  - data_arr[:, 2])

        # (max_arr - min_arr) / 10000

        if self.dtype == 'classifier':
            self.y_coords_log = np.log10(self.y_coords) / np.log10(self.y_base)
        else:
            self.y_coords_log = self.y_coords

        if self.dtype == 'doublet':
            self.ratio_color = self.y_coords/self.x_coords

        self.idx_current = None
        self.color_dict = cfg['colors']

        self.fig_format = parse_fig_cfg(fig_cfg, ax_diag, ax_line, self.dtype)
        self._fig, self._ax1, self._ax2 = None, None, None

        self.detection_range = np.linspace(data_arr[:,1].min(), data_arr[:,1].max(), 50) if self.dtype == 'classifier' else None

        return

    def show(self):

        # Generate the figure
        with rc_context(self.fig_format['fig']):

            # Create the figure
            self._fig, (self._ax1, self._ax2) = plt.subplots(1, 2)

            # Diagnostic plot
            scatter_plot(self._fig, self._ax1, self.x_coords, self.y_coords, self.id_arr, self.classes, self.color_dict,
                         idx_target=self.idx_current, detection_range=self.detection_range, ratio_color=self.ratio_color)
            ax_wording(self._ax1, self.fig_format['ax1'], legend_cfg={'loc': 'lower center', 'ncol':2, 'framealpha':0.95},
                       yscale=self.y_scale)

            # Line plot
            self.index_target()
            self.line_plot()
            ax_wording(self._ax2, self.fig_format['ax2'])

            # Interactive widget
            self._fig.canvas.mpl_connect('button_press_event', self._on_click)

            # Display the plot
            plt.tight_layout()
            plt.show()

        return

    def _on_click(self, event):

        if event.inaxes == self._ax1 and event.button == 1:

            if self.y_scale == 'log':
                user_point = (event.xdata, np.log10(event.ydata) / np.log10(self.y_base))
            else:
                user_point = (event.xdata, event.ydata)


            # Get index point
            self.index_target(user_point)

            # Replot the figures
            self._ax1.clear()
            scatter_plot(self._fig, self._ax1, self.x_coords, self.y_coords, self.id_arr, self.classes, self.color_dict,
                         idx_target=self.idx_current, detection_range=self.detection_range, ratio_color=self.ratio_color)
            ax_wording(self._ax1, self.fig_format['ax1'], legend_cfg={'loc': 'lower center', 'ncol':2, 'framealpha':0.95},
                       yscale=self.y_scale)

            self._ax2.clear()
            # self._fig.clear()

            self.line_plot()
            plt.tight_layout()
            self._fig.canvas.draw()

        return

    def index_target(self, mouse_coords=None):

        # If no selection use first point
        if mouse_coords is None:
            self.idx_current = 0
            print(f'Reseting location')

        else:
            distances = np.sqrt((self.x_coords - mouse_coords[0]) ** 2 + (self.y_coords_log - mouse_coords[1]) ** 2)
            self.idx_current = np.argmin(distances)
            print('Click on:', mouse_coords)#, self.ratio_color[self.idx_current])

        return

    def line_plot(self):

        feature = self.id_arr[self.idx_current]
        self._ax2.step(self.wave_range, self.data_df[self.idx_current, :], label=feature,
                      color=self.color_dict[feature], where='mid')

        return