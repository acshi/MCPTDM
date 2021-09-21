#!/bin/bash
mkdir -p figures/pdf

cd progressive_mcts
./make_all_figures.sh
cd ../

# Each separate invocation may be performed on a different machine
# and each of the results.cache files cat'ed together to improve the total execution time
# Even with 24 cores running in parallel, this still takes many hours to fully execute.

# We originally split all the work evenly in two to run on two machines
# by using the 0:2:16383 (all evens) and 1:2:16383 (all odds) syntax

time cargo run --release rng_seed 0:2:16383 :: method mpdm :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64
time cargo run --release rng_seed 0:2:16383 :: method eudm :: use_cfb false true :: eudm.samples_n 1 2 4 8 16 32
time cargo run --release rng_seed 0:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode classic :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0
time cargo run --release rng_seed 0:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0 32768

time cargo run --release rng_seed 1:2:16383 :: method mpdm :: use_cfb false :: mpdm.samples_n 2 4 8 16 32 64
time cargo run --release rng_seed 1:2:16383 :: method eudm :: use_cfb false true :: eudm.samples_n 1 2 4 8 16 32
time cargo run --release rng_seed 1:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode classic :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0
time cargo run --release rng_seed 1:2:16383 :: method mcts :: use_cfb false :: mcts.bound_mode marginal :: mcts.samples_n 8 16 32 64 128 256 :: mcts.repeat_const 0 32768

./plot.py final
