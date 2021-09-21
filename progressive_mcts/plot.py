#!/usr/bin/python3
import pdb
import sys
import math
import sqlite3
from common_plot import SqliteFigureBuilder, FigureMode

conn = sqlite3.connect("results.db")
db_cursor = conn.cursor()

show_only = False
make_pdf_also = False


t10s = dict()
t10s["regret"] = "Regret"
t10s["samples_n"] = "# Monte Carlo trials"
t10s["steps_taken"] = t10s["samples_n"] # as long as we do the proper rescaling!!!
t10s["bound_mode"] = "UCB expected-cost rule"
t10s["final_choice_mode"] = "Final choice expected-cost rule"
t10s["selection_mode"] = "UCB variation"
t10s["classic"] = "Classic"
t10s["lower_bound"] = "Lower bound"
t10s["expectimax"] = "Expectimax"
t10s["marginal"] = "MAC (proposed)"
t10s["ucb_const"] = "UCB constant factor"
t10s["ucb"] = "UCB"
t10s["ucbv"] = "UCB-V"
t10s["ucbd"] = "UCB-delta"
t10s["klucb"] = "KL-UCB"
t10s["klucb+"] = "KL-UCB+"
t10s["random"] = "Random"
t10s["uniform"] = "Uniform"
t10s["repeat_const"] = "Repetition constant"

figure_cmd_line_options = []
def should_make_figure(fig_name):
    figure_cmd_line_options.append(fig_name)
    return fig_name in sys.argv

all_metrics = ["regret"]

bound_mode = FigureMode("bound_mode", ["classic", "expectimax", "lower_bound", "marginal"])
ucb_const_vals = [0, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000]
ucb_const_mode = FigureMode("ucb_const", ucb_const_vals)
ucb_const_ticknames = [val / 10 for val in ucb_const_vals]
samples_n_mode = FigureMode("samples_n", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode classic expectimax lower_bound marginal :: final_choice_mode same :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
if should_make_figure("1"):
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)
        fig.plot(ucb_const_mode, [
                ("max.rng_seed", 511),
                ("selection_mode", "ucb"),
                ("repeat_const", -1),
                ("final_choice_mode", "same"),
        ], bound_mode)
        fig.ticks(ucb_const_ticknames)
        fig.legend()
        fig.show(title="Regret by UCB constant factor and expected-cost rule",
                 xlabel="UCB constant factor * 0.1",
                 file_suffix=f"_final_choice_mode_same")

# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal
# cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode classic expectimax lower_bound marginal :: final_choice_mode marginal :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
if should_make_figure("2"):
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        # samples_n = 128
        common_filters = [
            ("max.rng_seed", 511),
            ("final_choice_mode", "marginal"),
        ]
        uniform_filters = common_filters + [("selection_mode", "uniform")]
        fig.plot(ucb_const_mode,
                 common_filters + [("selection_mode", "ucb")], bound_mode)

        fig.line_from(uniform_filters, "Uniform")

        fig.ylim([10, 44])
        fig.ticks(ucb_const_ticknames)
        fig.legend()

        fig.show(xlabel="UCB constant factor * 0.1",
                 file_suffix=f"_final_choice_mode_marginal")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode classic expectimax lower_bound marginal :: final_choice_mode marginal :: classic.ucb_const -1000 :: expectimax.ucb_const -330 :: lower_bound.ucb_const -330 :: marginal.ucb_const -330
if should_make_figure("3"):
    for metric in all_metrics:
        common_filters = [
            ("max.rng_seed", 4095),
            ("final_choice_mode", "marginal"),
        ]
        filters = [("selection_mode", "ucb"),
                   ("classic.ucb_const", -1000),
                   ("expectimax.ucb_const", -330),
                   ("lower_bound.ucb_const", -330),
                   ("marginal.ucb_const", -330),
                   ] + common_filters
        uniform_filters = [("selection_mode", "uniform")] + common_filters

        fig = SqliteFigureBuilder(db_cursor, "steps_taken", metric, translations=t10s, x_param_scalar=0.25, x_param_log=True)

        fig.inset_plot([8.8, 12.2], [-0.5, 6], [0.4, 0.4, 0.57, 0.57])

        fig.plot(samples_n_mode, filters, bound_mode)
        fig.plot(samples_n_mode, uniform_filters, label="Uniform")

        fig.xlim([2.8, 12.4])
        fig.ticks(range(3, 12 + 1))
        fig.legend("lower left")
        fig.show(xlabel="log2(# of trials)")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode ucb ucbv ucbd klucb klucb+ uniform :: ucb.ucb_const -1000 :: ucbv.ucb_const -220 :: ucbv.ucbv_const 0.001 :: ucbd.ucb_const -100 :: ucbd.ucbd_const 0.1 :: klucb.ucb_const -0.1 :: klucb.klucb_max_cost 4700 :: klucb+.ucb_const -1 :: klucb+.klucb_max_cost 470
if should_make_figure("4"):
    for metric in all_metrics:
        selection_mode = FigureMode(
            "selection_mode", ["ucb", "ucbv", "ucbd", "klucb", "klucb+", "uniform"])
        fig = SqliteFigureBuilder(db_cursor, "steps_taken", metric, translations=t10s, x_param_scalar=0.25, x_param_log=True)
        filters = [
            ("max.rng_seed", 4095),
            ("bound_mode", "marginal"),
            ("repeat_const", -1),
            ("ucb.ucb_const", -1000),
            ("ucbv.ucb_const", -220),
            ("ucbv.ucbv_const", 0.001),
            ("ucbd.ucb_const", -100),
            ("ucbd.ucbd_const", 0.1),
            ("klucb.ucb_const", -0.1),
            ("klucb.klucb_max_cost", 4700),
            ("klucb+.ucb_const", -1.0),
            ("klucb+.klucb_max_cost", 470),
        ]

        fig.inset_plot([8.8, 12.2], [0.0, 3.5], [0.4, 0.4, 0.57, 0.57])

        fig.plot(samples_n_mode, filters, selection_mode)

        # fig.ylim([-20, 380])
        fig.xlim([2.8, 12.4])
        fig.ticks(range(3, 12 + 1))
        fig.legend("lower left")
        fig.show(xlabel="log2(# of trials)")

# cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 1024 :: repeat_const 0 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288
if should_make_figure("repeat_const"):
    # repeat_const_vals = [0, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
    repeat_const_vals = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)
    samples_n_mode = FigureMode("samples_n", [8, 16, 32, 64, 128, 256, 512, 1024]) #, 2048, 4096])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        filters = [
            ("max.rng_seed", 16383),
            ("selection_mode", "klucb"),
            ("klucb.ucb_const", -0.1),
            ("klucb.klucb_max_cost", 4700),
            ("bound_mode", "marginal"),
        ]

        fig.plot(repeat_const_mode, filters, samples_n_mode, normalize="first")
        fig.axhline(1, color="black")
        repeat_const_ticks = [math.log2(v) if v > 0 else "w/o" for v in repeat_const_vals]
        fig.ticks(repeat_const_ticks)
        # fig.ylim([0.86, 1.18])
        fig.legend("upper left")
        fig.show("Relative regret by repetition constant and # monte carlo trials", xlabel="log2(repetition constant)", ylabel="Relative regret, relative to w/o repetition")

# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode classic :: selection_mode ucb :: ucb_const -1000
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode classic
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096
# cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 65536
if should_make_figure("final"):
    repeat_const_vals = [0, 8, 16, 64, 128, 256, 512, 1024]
    repeat_const_mode = FigureMode("repeat_const", repeat_const_vals)
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, "steps_taken", metric, translations=t10s, x_param_scalar=0.25, x_param_log=True)

        common_filters = [
            ("max.rng_seed", 4095),
        ]

        normal_filters = common_filters + [
            ("selection_mode", "ucb"),
            ("ucb_const", -1000),
            ("bound_mode", "classic"),
            ("repeat_const", -1),
        ]
        klucb_filters = common_filters + [
            ("selection_mode", "klucb"),
            ("ucb_const", -0.1),
            ("klucb_max_cost", 4700),
            ("bound_mode", "classic"),
            ("repeat_const", -1),
        ]
        mac_filters = common_filters + [
            ("selection_mode", "klucb"),
            ("ucb_const", -0.1),
            ("klucb_max_cost", 4700),
            ("bound_mode", "marginal"),
            ("repeat_const", -1),
        ]
        final_filters = common_filters + [
            ("selection_mode", "klucb"),
            ("ucb_const", -0.1),
            ("klucb_max_cost", 4700),
            ("bound_mode", "marginal"),
            ("repeat_const", 65536),
        ]

        fig.plot(samples_n_mode, normal_filters, label="Classic")
        fig.plot(samples_n_mode, klucb_filters, label="also w/ KL-UCB")
        fig.plot(samples_n_mode, mac_filters, label="also w/ MAC (proposed)")
        fig.plot(samples_n_mode, final_filters, label="also w/ repetition (proposed)")

        fig.legend(title="Improvements")
        fig.ticks(range(3, 12 + 1))
        fig.show(xlabel="log2(# of trials)", file_suffix="_final_comparison")

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode ucbv :: ucbv.ucbv_const 0 0.0001 0.001 0.01 0.1 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
if should_make_figure("ucbv"):
    ucb_const_vals = [-10, -15, -22, -33, -47, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    ucbv_const_mode = FigureMode(
        "ucbv_const", [0, 0.0001, 0.001, 0.01, 0.1])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 127),
            ("selection_mode", "ucbv"),
            ("bound_mode", "marginal"),
        ], ucbv_const_mode)

        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.zoom(0.5)
        fig.show()

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode ucbd :: ucbd.ucbd_const 0.000001 0.00001 0.0001 0.001 0.01 0.1 1 :: ucb_const -10 -15 -22 -33 -47 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
if should_make_figure("ucbd"):
    ucb_const_vals = [-10, -15, -22, -33, -47, -68, -100, -150, -220, -330, -470, -680, -1000, -1500, -2200, -3300, -4700, -6800, -10000]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    ucbd_const_mode = FigureMode("ucbd_const", ["0.000001", "0.00001", 0.0001, 0.001, 0.01, 0.1, 1])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 4095),
            ("selection_mode", "ucbd"),
            ("bound_mode", "marginal"),
        ], ucbd_const_mode)

        fig.ticks(ucb_const_vals)
        fig.legend()
        fig.zoom(0.5)
        fig.show()

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode klucb :: klucb.klucb_max_cost 470 680 1000 1500 2200 3300 4700 6800 :: ucb_const -0.001 -0.0022 -0.0047 -0.01 -0.022 -0.047 -0.1 -0.22 -0.47 -1
if should_make_figure("klucb"):
    ucb_const_vals = [-0.001, -0.0022, -0.0047, -0.01, -0.022, -0.047, -0.1, -0.22, -0.47, -1]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    klucb_max_cost_mode = FigureMode("klucb_max_cost", [470, 680, 1000, 1500, 2200, 3300, 4700, 6800])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 511),
            ("selection_mode", "klucb"),
            ("bound_mode", "marginal"),
        ], klucb_max_cost_mode)

        fig.ticks(ucb_const_vals)
        fig.zoom(0.5)
        fig.legend()
        fig.show(file_suffix="_selection_mode_klucb")

# cargo run --release rng_seed 0-127 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode klucb+ :: klucb+.klucb_max_cost 470 680 1000 1500 2200 3300 4700 6800 :: ucb_const -0.001 -0.0022 -0.0047 -0.01 -0.022 -0.047 -0.1 -0.22 -0.47 -1
if should_make_figure("klucb+"):
    ucb_const_vals = [-0.001, -0.0022, -0.0047, -0.01, -0.022, -0.047, -0.1, -0.22, -0.47, -1]
    ucb_const_mode = FigureMode(
        "ucb_const", ucb_const_vals)
    klucb_max_cost_mode = FigureMode("klucb_max_cost", [470, 680, 1000, 1500, 2200, 3300, 4700, 6800])
    for metric in all_metrics:
        fig = SqliteFigureBuilder(db_cursor, None, metric, translations=t10s)

        fig.plot(ucb_const_mode, [
            ("max.rng_seed", 511),
            ("selection_mode", "klucb+"),
            ("bound_mode", "marginal"),
        ], klucb_max_cost_mode)

        fig.ticks(ucb_const_vals)
        fig.zoom(0.5)
        fig.legend()
        fig.show(file_suffix="_selection_mode_klucb+")

if len(sys.argv) == 1 or "help" in sys.argv:
    print("Valid figure options:")
    for option in figure_cmd_line_options:
        print(option)
