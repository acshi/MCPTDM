#!/usr/bin/python3
from common_plot import parse_parameters, FigureBuilder, FigureMode, print_all_parameter_values_used, evaluate_conditions, filter_extra
import time
import sys

t10s = dict()
t10s["discount_factor"] = "Discount Factor"
t10s["cost.safety"] = "Safety cost"
t10s["cost.efficiency"] = "Efficiency cost"
t10s["cost"] = "Cost"
t10s["efficiency"] = "Efficiency"
t10s["tree"] = "Tree"
t10s["mpdm"] = "MPDM"
t10s["eudm"] = "EUDM"
t10s["mcts"] = "MCPTDM (proposed)"
t10s["method"] = "Method"
t10s["false"] = "w/o CFB"
t10s["true"] = "CFB"
t10s["use_cfb"] = "CFB"
t10s["seconds"] = "Computation time (s)"
t10s["997_ts"] = "99.7% Computation time (s)"
t10s["95_ts"] = "95% Computation time (s)"
t10s["mean_ts"] = "Mean computation time (s)"
t10s["search_depth"] = "Search depth"
t10s["samples_n"] = "# Samples"
t10s["bound_mode"] = "UCB expected-cost rule"
t10s["final_choice_mode"] = "Final choice expected-cost rule"
t10s["selection_mode"] = "UCB variation"
t10s["classic"] = "Classic"
t10s["lower_bound"] = "Using lower bound"
t10s["expectimax"] = "Using expectimax"
t10s["marginal"] = "Using marginal action costs"
t10s["ucb_const"] = "UCB constant factor"

figure_cmd_line_options = []
def should_make_figure(fig_name):
    figure_cmd_line_options.append(fig_name)
    return fig_name in sys.argv

cache_file = sys.argv[2] if len(sys.argv) > 2 and ".cache" in sys.argv[2] else "results.cache"

start_time = time.time()
results = []
with open(cache_file, "r") as f:
    for line in f:
        parts = line.split()
        if len(parts) > 13:
            entry = dict()
            entry["params"] = parse_parameters(parts[0], skip=["search_depth", "total_forward_t", "max_steps", "safety_margin_low", "safety_margin_high", "accel", "steer"])
            entry["crashed"] = float(parts[5])
            entry["end_t"] = float(parts[6])
            entry["dist_travelled"] = float(parts[7])
            entry["efficiency"] = float(parts[8])
            entry["mean_ts"] = float(parts[9])
            entry["95_ts"] = float(parts[10])
            entry["997_ts"] = float(parts[11])
            entry["max_ts"] = float(parts[12])
            entry["stddev_ts"] = float(parts[13])

            entry["cost.efficiency"] = float(parts[1])
            entry["cost.safety"] = float(parts[2])
            entry["cost.accel"] = float(parts[3])
            entry["cost.steer"] = float(parts[4])
            entry["cost"] = entry["cost.efficiency"] + entry["cost.safety"] + \
                entry["cost.accel"] + entry["cost.steer"]

            results.append(entry)
        else:
            continue
print(f"took {time.time() - start_time:.2f} seconds to load data")

cfb_mode = FigureMode("use_cfb", ["false", "true"])

plot_metrics = ["cost", "cost.safety", "efficiency"]
evaluate_metrics = ["cost", "efficiency", "cost.efficiency",
                    "cost.safety", "cost.accel", "cost.steer", "seconds"]

# time cargo run --release rng_seed 0:2:511 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode ucb :: mcts.ucb_const -1e5 -2.2e5 -4.7e5 -1e6 -2.2e6 -4.7e6 -1e7 :: mcts.repeat_const 2048
# time cargo run --release rng_seed 1:2:511 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode ucb :: mcts.ucb_const -1e5 -2.2e5 -4.7e5 -1e6 -2.2e6 -4.7e6 -1e7 :: mcts.repeat_const 2048
if should_make_figure("ucb"):
    samples_n_vals = [8, 16, 32, 64, 128, 256]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    ucb_const_vals = [-1e5, -2.2e5, -4.7e5, -1e6, -2.2e6, -4.7e6, -1e7]
    ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)

    # repeat_mode = FigureMode("repeat_const", [-1, 2048])
    # bound_mode = FigureMode("bound_mode", ["marginal"])

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal"),
            ("selection_mode", "ucb"),
            ("min.samples_n", 256),
            ("max.rng_seed", 1023),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(ucb_const_mode, filters)
        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.show(file_suffix="_for_ucb")

# time cargo run --release rng_seed 0:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 2.2 3.3 4.7 10 22 47 :: mcts.ucb_const 1.5
# time cargo run --release rng_seed 1:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 2.2 3.3 4.7 10 22 47 :: mcts.ucb_const 1.5
# time cargo run --release rng_seed 0:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.68 1 1.5 2.2 3.3 4.7
# time cargo run --release rng_seed 1:2:1023 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.68 1 1.5 2.2 3.3 4.7
if should_make_figure("klucb"):
    samples_n_vals = [8, 16, 32, 64, 128, 256]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    klucb_max_cost_vals = [2.2, 3.3, 4.7, 10, 22, 47]
    klucb_max_cost_mode = FigureMode("klucb_max_cost", klucb_max_cost_vals)
    ucb_const_vals = [0.68, 1, 1.5, 2.2, 3.3, 4.7]
    ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)

    bound_mode = FigureMode("bound_mode", ["marginal"])

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal"),
            ("min.samples_n", 256),
            ("klucb_max_cost", 4.7),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(ucb_const_mode, filters)
        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal"),
            ("min.samples_n", 256),
            ("ucb_const", 1.5),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(klucb_max_cost_mode, filters)
        fig.ticks(klucb_max_cost_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

    if False:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal"),
            ("klucb_max_cost", 4.7),
            ("ucb_const", 1.5),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(samples_n_mode, filters)
        fig.ticks(samples_n_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

    if False:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal"),
            ("ucb_const", 0.47)
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(samples_n_mode, filters, klucb_max_cost_mode)
        fig.ticks(samples_n_vals)
        fig.legend()
        fig.show(file_suffix="_for_klucb")

# time cargo run --release rng_seed 0:2:16383 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 1.5 :: mcts.repeat_const 0 64 128 256 512 1024 2048 8192 32768
# time cargo run --release rng_seed 1:2:16383 :: method mcts :: mcts.samples_n 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 1.5 :: mcts.repeat_const 0 64 128 256 512 1024 2048 8192 32768
# time cargo run --release rng_seed 0:2:2047 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 0 64 128 256 512 1024 2048
# time cargo run --release rng_seed 1:2:2047 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 0 64 128 256 512 1024 2048
# time ../selfdriving rng_seed 0-4095 :: method mcts :: mcts.samples_n 8 16 32 64 128 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 4096 8192
# time ../selfdriving rng_seed 4096-8191 :: method mcts :: mcts.samples_n 128 256 :: mcts.bound_mode marginal :: mcts.selection_mode klucb :: mcts.klucb_max_cost 4.7 :: mcts.ucb_const 0.47 :: mcts.repeat_const 0 64 128 256 512 1024 2048 4096 8192
if should_make_figure("repeat"):
    samples_n_value = 8
    samples_n_vals = [samples_n_value] #[8, 16, 32, 64, 128, 256]
    samples_n_mode = FigureMode("samples_n", samples_n_vals)
    repeat_const_vals = [0, 64, 128, 256, 512, 1024, 2048, 8192, 32768]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)

    if True:
        filters = [
            ("method", "mcts"),
            ("bound_mode", "marginal"),
            ("klucb_max_cost", 4.7),
            ("ucb_const", 1.5),
        ]

        fig = FigureBuilder(results, None, "cost", translations=t10s)
        fig.plot(repeat_const_mode, filters, samples_n_mode) #, normalize="first")
        fig.line_from(filters + [("samples_n", samples_n_value), ("repeat_const", -1)], "old_repeat")
        fig.ticks(repeat_const_vals)
        fig.legend()
        fig.show(file_suffix=f"_{samples_n_value}")

# cargo run --release rng_seed 0:2:16383 :: method mpdm :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64
# cargo run --release rng_seed 0:2:16383 :: method eudm :: use_cfb false true :: eudm.samples_n 1 2 4 8 16 32
# cargo run --release rng_seed 0:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode classic :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0
# cargo run --release rng_seed 0:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0 32768
# cargo run --release rng_seed 1:2:16383 :: method mpdm :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64
# cargo run --release rng_seed 1:2:16383 :: method eudm :: use_cfb false true :: eudm.samples_n 1 2 4 8 16 32
# cargo run --release rng_seed 1:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode classic :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0
# cargo run --release rng_seed 1:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0 32768
if should_make_figure("final"):
    for do_ablation in [False, True]:
        for metric in ["cost.efficiency", "cost.safety", "cost", "efficiency"]:
            seconds_fig = FigureBuilder(results, "95_ts", metric, translations=t10s)
            common_filters = [("max.rng_seed", 16383), ("discount_factor", 0.8), ("safety", 600)]

            if not do_ablation:
                mpdm_filters = [("method", "mpdm"), ("use_cfb", "false")] + common_filters
                seconds_fig.plot(FigureMode("samples_n", [2, 4, 8, 16, 32, 64]),
                                 mpdm_filters, label="MPDM")

                eudm_filters = [("method", "eudm"),
                                ("allow_different_root_policy", "true")] + common_filters
                seconds_fig.plot(FigureMode("samples_n", [1, 2, 4, 8, 16, 32]), eudm_filters, cfb_mode, label="EUDM, ")

            if do_ablation:
                mcts_filters = [("method", "mcts"),
                                ("use_cfb", "false"),
                                ("repeat_const", 0),
                                ("selection_mode", "klucb"),
                                ("klucb_max_cost", 4.7),
                                ("ucb_const", 1.5),
                                ("bound_mode", "classic")] + common_filters
                seconds_fig.plot(FigureMode(
                    "samples_n", [8, 16, 32, 64, 128, 256]), mcts_filters, label="MCPTDM (-repeat, -MAC)")

                mcts_filters = [("method", "mcts"),
                                ("use_cfb", "false"),
                                ("repeat_const", 0),
                                ("selection_mode", "klucb"),
                                ("klucb_max_cost", 4.7),
                                ("ucb_const", 1.5),
                                ("bound_mode", "marginal")] + common_filters
                seconds_fig.plot(FigureMode(
                    "samples_n", [8, 16, 32, 64, 128, 256]), mcts_filters, label="MCPTDM (-repeat)")

            mcts_filters = [("method", "mcts"),
                            ("use_cfb", "false"),
                            ("repeat_const", 32768),
                            ("selection_mode", "klucb"),
                            ("klucb_max_cost", 4.7),
                            ("ucb_const", 1.5),
                            ("bound_mode", "marginal")] + common_filters
            seconds_fig.plot(FigureMode(
                "samples_n", [8, 16, 32, 64, 128, 256]), mcts_filters, label="MCPTDM (proposed)")

            seconds_fig.legend()
            metric_name = seconds_fig.translate(metric).lower()
            title = f"MCPTDM ablation: {metric_name} by 95% computation time (s)" if do_ablation else f"Final comparison: {metric_name} by 95% computation time (s)"
            seconds_fig.show(title=title, file_suffix="_ablation" if do_ablation else "_final")

if len(sys.argv) == 1 or "help" in sys.argv:
    print("Valid figure options:")
    for option in figure_cmd_line_options:
        print(option)
