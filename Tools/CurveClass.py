import pylab as pb
import numpy as np
import pandas as pd
import matplotlib as mpl


class CurveClass:
    def __init__(self, nb_curves=1, cmap_name='hot',
                 figsize=(15, 8), nrows=1, ncols=1,
                 axis_label_size=15, legend_size=22, title_size=22, ticks_size=15,
                 fig_ax=None):
        self.nb_curves = nb_curves
        self.cmap_name = cmap_name
        self.figsize = figsize
        self.nrows = nrows
        self.ncols = ncols
        self.axis_label_size = axis_label_size
        self.legend_size = legend_size
        self.title_size = title_size
        self.ticks_size = ticks_size
        self.ijs = [(i, j) for i in range(self.nrows) for j in range(self.ncols)]

        self.cmap = pb.cm.get_cmap(self.cmap_name)
        self.norm_color = mpl.colors.Normalize(0, self.nb_curves)

        if fig_ax is None:
            self.fig, self.ax = None, None
        else:
            self.fig, self.ax = fig_ax

        self.create_plot()

        if isinstance(self.ax, np.ndarray):
            self.curve_num = np.zeros(self.ax.shape)
        else:
            self.curve_num = 0

    def color(self, i):
        return self.cmap(self.norm_color(i))

    def create_plot(self):
        if self.fig is None:
            self.fig, self.ax = pb.subplots(self.nrows, self.ncols, figsize=self.figsize)

        self.__apply_on_all_axis(lambda ax: ax.tick_params(axis='both', labelsize=self.ticks_size))

    def plot(self, x, y, label=None, xlabel=None, ylabel=None, i=None, j=None, **kwargs):
        ax0 = self._get_ax(i, j)
        curve_num = self._get_curve_num(i, j)
        self._color_feature(curve_num, kwargs)
        self._set_plot_features(ax0, kwargs, xlabel, ylabel)
        ax0.plot(x, y, label=label, **kwargs)

    def plot_vertical_line(self, x, label=None, xlabel=None, ylabel=None, i=None, j=None, **kwargs):
        ax0 = self._get_ax(i, j)
        curve_num = self._get_curve_num(i, j)
        self._color_feature(curve_num, kwargs)
        self._set_plot_features(ax0, kwargs, xlabel, ylabel)
        ax0.axvline(x, label=label, **kwargs)

    def plot_horizontal_line(self, x, label=None, xlabel=None, ylabel=None, i=None, j=None, **kwargs):
        ax0 = self._get_ax(i, j)
        curve_num = self._get_curve_num(i, j)
        self._color_feature(curve_num, kwargs)
        self._set_plot_features(ax0, kwargs, xlabel, ylabel)
        ax0.axhline(x, label=label, **kwargs)

    def hist(self, vals, bins=10, label=None, xlabel=None, ylabel=None, i=None, j=None, normed=False, plot_zeros=False, **kwargs):
        ax0 = self._get_ax(i, j)
        curve_num = self._get_curve_num(i, j)
        self._color_feature(curve_num, kwargs)
        ax0, kwargs = self._set_plot_features(ax0, kwargs, xlabel, ylabel)
        
        vals_are_str = isinstance(vals[0], str)
        
        if vals_are_str:
            cat_names = list(set(vals))
            cat_name_dict = {cat_name: i for i, cat_name in enumerate(cat_names)}
            vals = pd.DataFrame(vals).replace(cat_name_dict)
            y, x = np.histogram(vals, bins=range(len(cat_names)+1))
            idx = np.argsort(y)[::-1]
            y = y[idx]
            cat_names = np.array(cat_names)[idx]
            x = x[:-1]
        else:
            y, x = np.histogram(vals, bins=bins)
            x = (x[1:] + x[:-1])/2.

        if normed == 'density':
            y = y / np.sum(y)
            y *= x[1]-x[0]
        elif normed == 'prop':
            y = y / np.sum(y)
            y /= np.sum(y)
        
        if plot_zeros is False and vals_are_str is False:
            mask = np.where(y != 0)[0]
            y = y[mask]
            x = x[mask]

        ax0.plot(x, y, label=label, **kwargs)
        
        if vals_are_str:
            ax0.set_xticks(range(len(cat_names)))
            ax0.set_xticklabels(cat_names, rotation=45, ha='right')
        
        return pd.DataFrame(list(zip(x, y)), columns=['x', 'y']).set_index('x')

    def colormap(self, tab, i=None, j=None, xlabel=None, ylabel=None, xtick_label=None, ytick_label=None,
                 xlabel_rotation=None, ylabel_rotation=None, **kwargs):
        ax0 = self._get_ax(i, j)
        ax0, kwargs = self._set_plot_features(ax0, kwargs, xlabel, ylabel)

        im = ax0.imshow(tab, cmap=self.cmap, **kwargs)
        pb.colorbar(im)

        ax0.set_xticks(np.arange(tab.shape[0]), xtick_label, rotation=xlabel_rotation, ha='right')
        ax0.set_yticks(np.arange(tab.shape[1]), ytick_label, rotation=ylabel_rotation, ha='right')

    def legend(self, i=None, j=None, **kwargs):
        ax0 = self._get_ax(i, j)

        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.legend_size

        ax0.legend(**kwargs)

    def title(self, title, i=None, j=None, **kwargs):
        ax0 = self._get_ax(i, j)

        if 'fontsize' not in kwargs:
            kwargs['fontsize'] = self.title_size

        ax0.set_title(title, **kwargs)

    def _set_plot_features(self, ax0, kwargs, xlabel, ylabel):
        if xlabel is not None:
            ax0.set_xlabel(xlabel, fontsize=self.axis_label_size)
        if ylabel is not None:
            ax0.set_ylabel(ylabel, fontsize=self.axis_label_size)

        return ax0, kwargs

    def _color_feature(self, curve_num, kwargs):
        if 'c' not in kwargs:
            kwargs['c'] = self.color(curve_num)

    def _get_ax(self, i, j):
        if i is None:
            ax0 = self.ax
        elif j is None:
            ax0 = self.ax[i]
        else:
            ax0 = self.ax[i, j]
        return ax0

    def _get_curve_num(self, i, j):
        if i is None:
            curve_num = self.curve_num
            self.curve_num += 1
        elif j is None:
            curve_num = self.curve_num[i]
            self.curve_num[i] += 1
        else:
            curve_num = self.curve_num[i, j]
            self.curve_num[i, j] += 1
        return curve_num

    def __apply_on_all_axis(self, fct):
        if not(isinstance(self.ax, np.ndarray)):
            fct(self.ax)
        elif not(isinstance(self.ax[0], np.ndarray)):
            for ax0 in self.ax:
                fct(ax0)
        else:
            for axs in self.ax:
                for ax0 in axs:
                    fct(ax0)
