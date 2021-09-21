#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
import numbers

plt.rcParams.update({'font.size': 12})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["axes.prop_cycle"] = (
    plt.cycler(
        color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#666666"] * 2) +
    plt.cycler(
        marker=["+", "o", "x", "^", "v", "s", "*", "D"] * 2) + plt.cycler(linestyle=["-"] * 6 + ["--"] * 6 + ["-."] * 4))


show_only = False
make_pdf_also = True

save_dpi = 300
figure_zoom = 1.25


class FigureMode:
    def __init__(self, param, values):
        self.param = param
        self.values = values

    def matches(self, params, value):
        str_value = str(value)
        if str_value.endswith(".0"):
            str_value = str_value.replace(".0", "")
        return self.param in params and params[self.param] == str_value


def filter_match(params, filter):
    param = filter[0]
    param_value = filter[1]
    param_split = param.split(".")

    if len(param_split) == 2:
        if param_split[0] == "max":
            name_value = float(params[param_split[1]])
            return name_value <= param_value
        elif param_split[0] == "min":
            name_value = float(params[param_split[1]])
            return name_value >= param_value
        elif params["method"] != param_split[0]:
            return True
        param = param_split[1]

    return param in params and params[param] == str(param_value)


def filter_extra(results, filters):
    return [entry for entry in results if all(filter_match(entry["params"], f) for f in filters)]


def short_num_string(val):
    scientific = f"{val:.1e}".replace(".0e", "e").replace(
        "+", "").replace("e0", "e").replace("e-0", "e-")
    normal = str(val)
    # same-length means scientific is shorter in most fonts because of "." being short
    if len(scientific) == len(normal) and "." in scientific:
        return scientific
    return scientific if len(scientific) < len(normal) else normal


def decapitalize_word(word):
    if len(word) == 0:
        return word
    if word.isupper():
        return word
    return word[0].lower() + word[1:]


def decapitalize(title):
    words = title.split(" ")
    return " ".join(decapitalize_word(word) for word in words)

class FigureBuilder:
    def __init__(self, results, x_param, y_param, translations={}):
        self.results = results
        self.x_param = x_param
        self.defacto_x_param = x_param
        self.y_param = y_param
        self.all_modes = []
        self.translations = translations
        self.min_x = None
        self.max_x = None
        self.x_locs = []
        self.axins = None
        self.figure_zoom = 1

        self.fig, self.ax = plt.subplots(dpi=100 if show_only else save_dpi)

    def translate(self, name):
        if name in self.translations:
            return self.translations[name]
        return name

    def filter_entry(self, entry, filters, modes=[]):
        params = entry["params"]
        return self.y_param in entry and all(filter_match(params, f) for f in filters) and all(mode_val[0].matches(params, mode_val[1]) for mode_val in modes)

    def collect_vals(self, x_mode, filters, legend_mode, legend_mode_val):
        x_val_sets = [list() for _ in range(len(x_mode.values))]
        y_val_sets = [list() for _ in range(len(x_mode.values))]
        modes = [(legend_mode, legend_mode_val)] if legend_mode else []
        for entry in self.results:
            if not self.filter_entry(entry, filters, modes):
                continue
            for (i, val_name) in enumerate(x_mode.values):
                # if entry["params"][x_mode.param] == str(val_name):
                if x_mode.matches(entry["params"], val_name):
                    if self.x_param is not None:
                        x_val_sets[i].append((entry[self.x_param])
                                             if self.x_param in entry else float(entry["params"][self.x_param]))
                    y_val_sets[i].append((entry[self.y_param]))
        return (x_val_sets, y_val_sets)

    def plot(self, x_mode, filters=[], legend_mode=None, label=None, normalize=None):
        if self.defacto_x_param is None:
            self.defacto_x_param = x_mode.param

        if legend_mode:
            if not any(legend_mode.param == mode.param for mode in self.all_modes):
                self.all_modes += [legend_mode]

        for legend_mode_val in legend_mode.values if legend_mode else [None]:
            import time
            start_time = time.time()
            (x_val_sets, y_val_sets) = self.collect_vals(
                x_mode, filters, legend_mode, legend_mode_val)
            print(f"collect_vals took {time.time() - start_time:.2} seconds")
            if len(y_val_sets) == 0:
                label_str = f"{label}: " if label else ""
                print(
                    f"{label_str}Data completely missing for {self.y_param} by {x_mode.param} with {filters}")
                if legend_mode:
                    print(f"And with {legend_mode.param} = {legend_mode_val}")
                continue
            n_vals_in_set = len(y_val_sets[0])
            for i, vals in enumerate(y_val_sets):
                if len(vals) == 0:
                    label_str = f"{label}: " if label else ""
                    legend_str = f"and with {legend_mode.param} = {legend_mode_val}" if legend_mode else ""
                    print(
                        f"{label_str}{x_mode.param} = {x_mode.values[i]} has 0 data points for {self.y_param} with {filters} {legend_str}")
                    vals.append(np.nan)
                if len(vals) != n_vals_in_set:
                    label_str = f"{label}: " if label else ""
                    legend_str = f"and with {legend_mode.param} = {legend_mode_val}" if legend_mode else ""
                    print(
                        f"{label_str}{len(vals)} != {n_vals_in_set} for {x_mode.param} = {x_mode.values[i]} {legend_str}")

            means = np.array([np.mean(vals) for vals in y_val_sets])
            stdev_mean = np.array([np.std(vals) / np.sqrt(len(vals))
                          for vals in y_val_sets])

            if self.x_param is None:
                x_means = np.array([i for i in range(len(x_val_sets))])
            else:
                x_means = np.array([np.mean(vals) for vals in x_val_sets])
            self.x_locs = x_means

            full_label = label
            if legend_mode:
                if full_label:
                    full_label = f"{label} {self.translate(legend_mode_val)}"
                else:
                    full_label = f"{self.translate(legend_mode_val)}"

            if normalize == "last":
                factor = means[-1]
                means /= factor
                stdev_mean /= factor
            elif normalize == "first":
                factor = means[0]
                means /= factor
                stdev_mean /= factor

            self.ax.errorbar(x_means, means,
                             yerr=stdev_mean, label=full_label)

            if self.axins:
                self.axins.errorbar(x_means, means, yerr=stdev_mean)

            x_mean_min = np.min(x_means)
            if self.min_x is None:
                self.min_x = x_mean_min
            else:
                self.min_x = min(self.min_x, x_mean_min)

            x_mean_max = np.max(x_means)
            if self.max_x is None:
                self.max_x = x_mean_max
            else:
                self.max_x = max(self.max_x, x_mean_max)

    def line_from(self, filters, label):
        vals = [entry[self.y_param] for entry in filter_extra(self.results, filters)]
        mean = np.mean(vals)
        stdev_mean = np.std(vals) / np.sqrt(len(vals))
        self.ax.errorbar([self.min_x, self.max_x], [mean, mean],
                         yerr=[stdev_mean, stdev_mean], label=label)
        if self.axins:
            self.axins.errorbar([self.min_x, self.max_x], [mean, mean],
                                yerr=[stdev_mean, stdev_mean])

    def axhline(self, y, **kwargs):
        self.ax.axhline(y, **kwargs)

    def _set_show_save(self, title, xlabel, ylabel, file_suffix):
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if show_only:
            plt.show()
        else:
            self.fig.set_figwidth(6.4 * figure_zoom)
            self.fig.set_figheight(4.8 * figure_zoom)

            modes_description = "_".join([""] + [mode.param for mode in self.all_modes])
            file_desc = f"{self.defacto_x_param}_{self.y_param}{modes_description}{file_suffix}"

            self.fig.tight_layout()
            if make_pdf_also:
                self.fig.savefig(f"figures/pdf/by_{file_desc}.pdf",
                                 bbox_inches="tight", pad_inches=0)
            self.fig.savefig(f"figures/by_{file_desc}.png")

    def show(self, title=None, xlabel=None, ylabel=None, file_suffix=""):
        xlabel = xlabel or self.translate(self.defacto_x_param)
        ylabel = ylabel or self.translate(self.y_param)

        if title is None:
            title = f"{self.translate(self.y_param)} by {decapitalize(self.translate(self.defacto_x_param))}"

            modes_str = " and ".join([""] + [decapitalize(self.translate(mode.param))
                                             for mode in self.all_modes])
            title += modes_str

        modes_str = " and ".join([""] + [mode.param for mode in self.all_modes])
        print(f"{self.y_param} by {self.defacto_x_param}{modes_str}")

        self._set_show_save(title, xlabel, ylabel, file_suffix)

    def legend(self, loc=None):
        self.ax.legend(loc=loc)

    def xlim(self, xlim):
        self.ax.set_xlim(xlim)

    def ylim(self, ylim):
        self.ax.set_ylim(ylim)

    def xscale(self, xscale):
        self.ax.set_xscale(xscale)

    def yscale(self, yscale):
        self.ax.set_yscale(yscale)

    def ticks(self, labels, locs=None):
        if any(isinstance(val, numbers.Number) and abs(val) >= 1e5 for val in labels):
            labels = [short_num_string(val) for val in labels]

        if locs is None:
            locs = self.x_locs

        self.ax.set_xticks(locs)
        self.ax.set_xticklabels(labels)

    def ax(self):
        return self.ax

    def fig(self):
        return self.fig

    # Call this before any plotting to also have things plot in the inset
    def inset_plot(self, xlim, ylim, bounds=[0.5, 0.5, 0.47, 0.47]):
        self.axins = self.ax.inset_axes(bounds)

        self.axins.set_xlim(xlim)
        self.axins.set_ylim(ylim)
        self.axins.set_xticklabels("")
        self.axins.set_yticklabels("")

        return self.ax.indicate_inset_zoom(self.axins, edgecolor="black")

def single_where_clause(param, value):
    if param.startswith("max."):
        param = param.split("max.")[1]
        return f"{param} <= '{value}'"
    elif "." in param:
        parts = param.split(".")
        (filter_mode_val, param_name) = (parts[0], parts[1])
        if filter_mode_val in ["classic", "expectimax", "lower_bound", "marginal", "marginal_prior"]:
            filter_mode = "bound_mode"
        elif filter_mode_val in ["ucb", "ucbv", "ucbd", "klucb", "klucb+", "uniform"]:
            filter_mode = "selection_mode"
        else:
            print(f"don't recognize '{filter_mode_val}' for filter modes")
            exit(1)
        return f"({param_name} = '{value}' OR {filter_mode} != '{filter_mode_val}')"
    return f"{param} = '{value}'"

def where_clause(filters, modes):
    return "WHERE " + " AND ".join([single_where_clause(p, v) for (p, v) in filters] + [f"{mode.param} = '{v}'" for (mode, v) in modes])

def db_filtered(db_cursor, param, filters):
    return [val for val in db_cursor.execute(f"SELECT {param} FROM results {where_clause(filters, [])}")]

class SqliteFigureBuilder:
    def __init__(self, db_cursor, x_param, y_param, translations={}, x_param_scalar=1, x_param_log=False):
        self.db_cursor = db_cursor
        self.x_param = x_param
        self.defacto_x_param = x_param
        self.y_param = y_param
        self.all_modes = []
        self.translations = translations
        self.x_param_log = x_param_log
        self.x_param_scalar = x_param_scalar
        self.min_x = None
        self.max_x = None
        self.x_locs = []
        self.axins = None
        self.figure_zoom = 1
        self.figure_height_scale = 1

        self.fig, self.ax = plt.subplots(dpi=100 if show_only else save_dpi)

    def translate(self, name):
        if name in self.translations:
            return self.translations[name]
        return name

    def collect_vals(self, x_mode, filters, legend_mode, legend_mode_val):
        modes = [(legend_mode, legend_mode_val)] if legend_mode else []

        x_val_sets_raw = {}
        y_val_sets_raw = {}

        # x_param = self.x_param if self.x_param else x_mode.param
        # select_sql = f"SELECT {x_param}, {self.y_param} FROM results {where_clause(filters, modes)}"
        # print(select_sql)
        # for (x_val, y_val) in self.db_cursor.execute(select_sql):
        #     if x_val not in x_val_sets_raw:
        #         x_val_sets_raw[x_val] = []
        #         y_val_sets_raw[x_val] = []
        #     x_val_sets_raw[x_val].append(float(x_val))
        #     y_val_sets_raw[x_val].append(float(y_val))

        if self.x_param:
            select_sql = f"SELECT {self.x_param}, {x_mode.param}, {self.y_param} FROM results {where_clause(filters, modes)}"
            # print(select_sql)
            for (x_val, x_mode_val, y_val) in self.db_cursor.execute(select_sql):
                if x_mode_val not in x_val_sets_raw:
                    x_val_sets_raw[x_mode_val] = []
                    y_val_sets_raw[x_mode_val] = []
                x_val = float(x_val) * self.x_param_scalar
                x_val = np.log2(x_val) if self.x_param_log else x_val
                x_val_sets_raw[x_mode_val].append(float(x_val))
                y_val_sets_raw[x_mode_val].append(float(y_val))
        else:
            select_sql = f"SELECT {x_mode.param}, {self.y_param} FROM results {where_clause(filters, modes)}"
            # print(select_sql)
            for (x_mode_val, y_val) in self.db_cursor.execute(select_sql):
                x_val = x_mode_val
                if x_mode_val not in x_val_sets_raw:
                    x_val_sets_raw[x_mode_val] = []
                    y_val_sets_raw[x_mode_val] = []
                x_val = float(x_val) * self.x_param_scalar
                x_val = np.log2(x_val) if self.x_param_log else x_val
                x_val_sets_raw[x_mode_val].append(float(x_val))
                y_val_sets_raw[x_mode_val].append(float(y_val))

        x_val_sets = [list() for _ in range(len(x_mode.values))]
        y_val_sets = [list() for _ in range(len(x_mode.values))]

        for x_val in x_val_sets_raw:
            is_number = isinstance(x_val, numbers.Number)
            for (i, val_name) in enumerate(x_mode.values):
                if is_number and float(x_val) == float(val_name) or not is_number and x_val == str(val_name):
                    x_val_sets[i] = x_val_sets_raw[x_val]
                    y_val_sets[i] = y_val_sets_raw[x_val]

        return (x_val_sets, y_val_sets)

    def plot(self, x_mode, filters=[], legend_mode=None, label=None, normalize=None):
        if self.defacto_x_param is None:
            self.defacto_x_param = x_mode.param

        if legend_mode:
            if not any(legend_mode.param == mode.param for mode in self.all_modes):
                self.all_modes += [legend_mode]

        for legend_mode_val in legend_mode.values if legend_mode else [None]:
            import time
            start_time = time.time()
            (x_val_sets, y_val_sets) = self.collect_vals(
                x_mode, filters, legend_mode, legend_mode_val)
            print(f"collect_vals took {time.time() - start_time:.2} seconds")
            if len(y_val_sets) == 0:
                label_str = f"{label}: " if label else ""
                print(
                    f"{label_str}Data completely missing for {self.y_param} by {x_mode.param} with {filters}")
                if legend_mode:
                    print(f"And with {legend_mode.param} = {legend_mode_val}")
                continue
            n_vals_in_set = len(y_val_sets[0])
            for i, vals in enumerate(y_val_sets):
                if len(vals) == 0:
                    label_str = f"{label}: " if label else ""
                    legend_str = f"and with {legend_mode.param} = {legend_mode_val}" if legend_mode else ""
                    print(
                        f"{label_str}{x_mode.param} = {x_mode.values[i]} has 0 data points for {self.y_param} with {filters} {legend_str}")
                    vals.append(np.nan)
                if len(vals) != n_vals_in_set:
                    label_str = f"{label}: " if label else ""
                    legend_str = f"and with {legend_mode.param} = {legend_mode_val}" if legend_mode else ""
                    print(
                        f"{label_str}{len(vals)} != {n_vals_in_set} for {x_mode.param} = {x_mode.values[i]} {legend_str}")

            means = np.array([np.mean(vals) for vals in y_val_sets])
            stdev_mean = np.array([np.std(vals) / np.sqrt(len(vals))
                          for vals in y_val_sets])

            if self.x_param is None:
                x_means = np.array([i for i in range(len(x_val_sets))])
            else:
                x_means = np.array([np.mean(vals) for vals in x_val_sets])
            self.x_locs = x_means

            full_label = label
            if legend_mode:
                if full_label:
                    full_label = f"{label} {self.translate(legend_mode_val)}"
                else:
                    full_label = f"{self.translate(legend_mode_val)}"

            if normalize == "last":
                factor = means[-1]
                means /= factor
                stdev_mean /= factor
            elif normalize == "first":
                factor = means[0]
                means /= factor
                stdev_mean /= factor

            self.ax.errorbar(x_means, means,
                             yerr=stdev_mean, label=full_label)

            if self.axins:
                self.axins.errorbar(x_means, means, yerr=stdev_mean)

            x_mean_min = np.min(x_means)
            if self.min_x is None:
                self.min_x = x_mean_min
            else:
                self.min_x = min(self.min_x, x_mean_min)

            x_mean_max = np.max(x_means)
            if self.max_x is None:
                self.max_x = x_mean_max
            else:
                self.max_x = max(self.max_x, x_mean_max)

    def line_from(self, filters, label):
        vals = db_filtered(self.db_cursor, self.y_param, filters)
        mean = np.mean(vals)
        stdev_mean = np.std(vals) / np.sqrt(len(vals))
        self.ax.errorbar([self.min_x, self.max_x], [mean, mean],
                         yerr=[stdev_mean, stdev_mean], label=label)
        if self.axins:
            self.axins.errorbar([self.min_x, self.max_x], [mean, mean],
                                yerr=[stdev_mean, stdev_mean])

    def axhline(self, y, **kwargs):
        self.ax.axhline(y, **kwargs)

    def _set_show_save(self, title, xlabel, ylabel, file_suffix):
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if show_only:
            plt.show()
        else:
            self.fig.set_figwidth(6.4 * figure_zoom * self.figure_zoom)
            self.fig.set_figheight(4.8 * figure_zoom * self.figure_zoom * self.figure_height_scale)

            modes_description = "_".join([""] + [mode.param for mode in self.all_modes])
            file_desc = f"{self.defacto_x_param}_{self.y_param}{modes_description}{file_suffix}"

            self.fig.tight_layout()
            if make_pdf_also:
                self.fig.savefig(f"figures/pdf/by_{file_desc}.pdf",
                                 bbox_inches="tight", pad_inches=0)
            self.fig.savefig(f"figures/by_{file_desc}.png")

    def show(self, title=None, xlabel=None, ylabel=None, file_suffix=""):
        xlabel = xlabel or self.translate(self.defacto_x_param)
        ylabel = ylabel or self.translate(self.y_param)

        if title is None:
            title = f"{self.translate(self.y_param)} by {decapitalize(self.translate(self.defacto_x_param))}"

            modes_str = " and ".join([""] + [decapitalize(self.translate(mode.param))
                                             for mode in self.all_modes])
            title += modes_str

        modes_str = " and ".join([""] + [mode.param for mode in self.all_modes])
        print(f"{self.y_param} by {self.defacto_x_param}{modes_str}")

        self._set_show_save(title, xlabel, ylabel, file_suffix)

    def legend(self, loc=None, title=None):
        if title is None and len(self.all_modes) > 0:
            title = self.translate(self.all_modes[0].param)
        self.ax.legend(loc=loc, title=title)

    def xlim(self, xlim):
        self.ax.set_xlim(xlim)

    def ylim(self, ylim):
        self.ax.set_ylim(ylim)

    def xscale(self, xscale):
        self.ax.set_xscale(xscale)
        if self.axins:
            self.axins.set_xscale(xscale)

    def yscale(self, yscale):
        self.ax.set_yscale(yscale)
        if self.axins:
            self.axins.set_yscale(yscale)

    def zoom(self, zoom):
        self.figure_zoom = 1.0 / zoom

    def height_scale(self, scale):
        self.figure_height_scale = scale

    def ticks(self, labels, locs=None):
        if any(isinstance(val, numbers.Number) and abs(val) >= 1e5 for val in labels):
            labels = [short_num_string(val) for val in labels]

        labels = [str(l) for l in labels]
        labels = [l.replace(".0", "") if l.endswith(".0") else l for l in labels]

        if locs is None:
            locs = self.x_locs

        self.ax.set_xticks(locs)
        self.ax.set_xticklabels(labels)

    def ax(self):
        return self.ax

    def fig(self):
        return self.fig

    # Call this before any plotting to also have things plot in the inset
    def inset_plot(self, xlim, ylim, bounds=[0.5, 0.5, 0.47, 0.47]):
        self.axins = self.ax.inset_axes(bounds)

        self.axins.set_xlim(xlim)
        self.axins.set_ylim(ylim)
        self.axins.set_xticklabels("")
        self.axins.set_yticklabels("")

        (rect, connections) = self.ax.indicate_inset_zoom(self.axins, edgecolor="black")

        for connection in connections:
            connection.set_visible(False)
        rect.set_label(None)

        return (rect, connections)


def evaluate_conditions(results, metrics, filters):
    results = filter_extra(results, filters)
    filters_string = ",".join([f"{f[0]}={f[1]}" for f in filters])

    print(f"{filters_string}:")

    return_results = []

    for metric in metrics:
        vals = [entry[metric] for entry in results]
        mean = np.mean(vals)
        stdev_mean = np.std(vals) / np.sqrt(len(vals))
        print(f"  {metric} has mean: {mean:6.4} and mean std dev: {stdev_mean:6.4} and a total of {len(vals)} samples")
        return_results.append(mean)
    print()

    return return_results


def print_all_parameter_values_used(results, filters):
    param_sets = {}
    for result in filter_extra(results, filters):
        for param_name in result["params"]:
            param_value = result["params"][param_name]
            if param_name not in param_sets:
                param_sets[param_name] = {}
            param_set = param_sets[param_name]
            if param_value not in param_set:
                param_set[param_value] = 0
            param_set[param_value] += 1
    for param_name in param_sets:
        param_set = param_sets[param_name]

        if param_name == "rng_seed":
            max_seed = max(int(val) for val in param_set)
            print(f"maximum rng_seed: {max_seed}")
            continue

        print(f"{param_name} has values: " +
              ", ".join(f"({param_value}: {param_set[param_value]})" for param_value in param_set))


def parse_parameters(parameters_string, skip=[]):
    parsed_params = {}
    for param in parameters_string.split(","):
        if len(param) == 0:
            continue
        param_split = param.split("=")
        param_name = param_split[0]
        param_value = param_split[1]
        if param_name not in skip:
            parsed_params[param_name] = param_value
    return parsed_params
