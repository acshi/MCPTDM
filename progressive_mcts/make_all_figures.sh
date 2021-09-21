#!/bin/bash
mkdir -p figures/pdf
./make_expected_cost_figure.py

time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode classic expectimax lower_bound marginal :: final_choice_mode same :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
./plot.py 1

time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal
time cargo run --release rng_seed 0-511 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode classic expectimax lower_bound marginal :: final_choice_mode marginal :: ucb_const 0 -68 -100 -150 -220 -330 -470 -680 -1000 -1500 -2200 -3300 -4700 -6800 -10000
./plot.py 2

time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode uniform :: final_choice_mode marginal
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: selection_mode ucb :: bound_mode classic expectimax lower_bound marginal :: final_choice_mode marginal :: classic.ucb_const -1000 :: expectimax.ucb_const -330 :: lower_bound.ucb_const -330 :: marginal.ucb_const -330
./plot.py 3

time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode marginal :: selection_mode ucb ucbv ucbd klucb klucb+ uniform :: ucb.ucb_const -1000 :: ucbv.ucb_const -220 :: ucbv.ucbv_const 0.001 :: ucbd.ucb_const -100 :: ucbd.ucbd_const 0.1 :: klucb.ucb_const -0.1 :: klucb.klucb_max_cost 4700 :: klucb+.ucb_const -1 :: klucb+.klucb_max_cost 470
./plot.py 4

time cargo run --release rng_seed 0-16383 :: samples_n 8 16 32 64 128 256 512 1024 :: repeat_const 0 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288
./plot.py repeat_const

time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode classic :: selection_mode ucb :: ucb_const -1000
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: bound_mode classic
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096
time cargo run --release rng_seed 0-4095 :: samples_n 8 16 32 64 128 256 512 1024 2048 4096 :: repeat_const 65536
./plot.py final
