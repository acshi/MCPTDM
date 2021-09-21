use std::{
    f64::consts::PI,
    rc::Rc,
    time::{Duration, Instant},
};

use arg_parameters::Parameters;

use cfb::conditional_focused_branching;
use mpdm::{make_obstacle_vehicle_policy_choices, mpdm_choose_policy};

use cost::Cost;
use rand::{prelude::StdRng, Rng, SeedableRng};
use rate_timer::RateTimer;
use reward::Reward;
use road::Road;
use road_set::RoadSet;
use rvx::{Rvx, RvxColor};

use crate::{eudm::dcp_tree_choose_policy, mcts::mcts_choose_policy};

#[macro_use]
extern crate fstrings;

mod arg_parameters;
mod belief;
mod car;
mod cfb;
mod cost;
mod delayed_policy;
mod eudm;
mod forward_control;
mod intelligent_driver;
mod lane_change_policy;
mod mcts;
mod mpdm;
mod open_loop_policy;
mod pure_pursuit;
mod rate_timer;
mod reward;
mod road;
mod road_set;
mod side_control;
mod side_policies;

#[macro_use]
extern crate enum_dispatch;

const AHEAD_TIME_DEFAULT: f64 = 0.6;

struct State {
    scenario_rng: StdRng,
    respawn_rng: StdRng,
    policy_rng: StdRng,
    params: Rc<Parameters>,
    road: Road,
    traces: Vec<rvx::Shape>,
    r: Option<Rvx>,
    timesteps: u32,
    reward: Reward,
    paper_graphics_sets: Vec<Vec<rvx::Shape>>,
}

impl State {
    fn update_graphics(&mut self) {
        if let Some(r) = self.r.as_mut() {
            r.clear();

            self.road.draw(r);
            r.draw_all(self.traces.iter().cloned());

            if self.params.graphics_for_paper && self.timesteps >= 1100 && self.timesteps % 50 == 25
            {
                self.paper_graphics_sets.push(r.shapes().to_vec());
            }

            r.set_global_rot(-PI / 2.0);
            r.commit_changes();
        }
    }

    fn update(&mut self, dt: f64) {
        let replan_interval = (self.params.replan_dt / self.params.physics_dt).round() as u32;

        // method chooses the ego policy
        let policy_rng = &mut self.policy_rng;
        if self.timesteps % replan_interval == 0 && !self.road.cars[0].crashed {
            let replan_real_time_start = Instant::now();

            let (policy, traces) = match self.params.method.as_str() {
                "fixed" => (None, Vec::new()),
                "mpdm" => mpdm_choose_policy(&self.params, &self.road, policy_rng),
                "eudm" => dcp_tree_choose_policy(&self.params, &self.road, policy_rng),
                "mcts" => mcts_choose_policy(&self.params, &self.road, policy_rng),
                _ => panic!("invalid method '{}'", self.params.method),
            };

            self.reward
                .planning_times
                .push(replan_real_time_start.elapsed().as_secs_f64());

            self.traces = traces;

            if let Some(policy) = policy {
                self.road.set_ego_policy(policy);
            }
        }

        // random policy changes for the obstacle vehicles
        let policy_change_interval =
            (self.params.nonego_policy_change_dt / self.params.physics_dt).round() as u32;
        let timesteps = self.timesteps;
        if self.timesteps % policy_change_interval == 0 {
            let rng = &mut self.scenario_rng;
            let policy_choices = make_obstacle_vehicle_policy_choices(&self.params);

            for c in self.road.cars[1..].iter_mut() {
                if rng.gen_bool(
                    self.params.nonego_policy_change_prob * self.params.nonego_policy_change_dt,
                ) {
                    let new_policy_i = rng.gen_range(0..policy_choices.len());
                    let new_policy = policy_choices[new_policy_i].clone();

                    if self.road.debug && self.params.obstacle_car_debug {
                        eprintln_f!("{timesteps}: obstacle car {c.car_i} switching to policy {new_policy_i}: {new_policy:?}");
                    }

                    c.side_policy = Some(new_policy);
                }
            }
        }

        // actual simulation
        self.road.update_belief();
        self.road.update(dt);
        self.road.respawn_obstacle_cars(&mut self.respawn_rng);

        // final reporting reward (separate from cost function, though similar)
        self.reward.dist_travelled += self.road.cars[0].vel * dt;
        if self.road.cars[0].crashed {
            self.reward.crashed = true;
        }

        self.timesteps += 1;
    }
}

fn run_with_parameters(params: Parameters) -> (Cost, Reward) {
    let params = Rc::new(params);

    let mut full_seed = [0; 32];
    full_seed[0..8].copy_from_slice(&params.rng_seed.to_le_bytes());

    let mut scenario_rng = StdRng::from_seed(full_seed);

    let mut road = Road::new(params.clone());
    // road.add_obstacle(100.0, 0);
    while road.cars.len() < params.n_cars + 1 {
        road.add_random_car(&mut scenario_rng);
    }
    road.init_belief();

    let mut state = State {
        scenario_rng,
        respawn_rng: StdRng::from_seed(full_seed),
        policy_rng: StdRng::from_seed(full_seed),
        road,
        r: None,
        timesteps: 0,
        params,
        traces: Vec::new(),
        reward: Default::default(),
        paper_graphics_sets: Vec::new(),
    };

    let use_graphics = !state.params.run_fast;

    if use_graphics {
        let mut r = Rvx::new("Self-Driving!", [0, 0, 0, 0], 8000);
        // r.set_user_zoom(Some(0.4)); // 0.22
        std::thread::sleep(Duration::from_millis(500));
        r.set_user_zoom(None);
        state.r = Some(r);
    }

    let mut rate = RateTimer::new(Duration::from_millis(
        (state.params.physics_dt * 1000.0 / state.params.graphics_speedup) as u64,
    ));

    for _ in 0..state.params.max_steps {
        state.update(state.params.physics_dt);

        if use_graphics {
            state.update_graphics();
            rate.wait_until_ready();
        }

        // if i == 1000 {
        //     for side_policy in state.road.cars[0].side_policy.iter_mut() {
        //         *side_policy = side_policies::SidePolicy::LaneChangePolicy(
        //             lane_change_policy::LaneChangePolicy::new(1, LANE_CHANGE_TIME, None),
        //         );
        //     }
        // }
    }

    if state.params.graphics_for_paper {
        if let Some(r) = state.r.as_mut() {
            r.clear();
            r.draw(Rvx::square().scale(1000.0).color(RvxColor::LIGHT_GRAY));

            let x = 0.0;
            let mut y = 0.0;

            for shape_set in state.paper_graphics_sets.iter() {
                r.draw_all(shape_set.iter().cloned());
                r.set_translate_modifier(x, y);
                y -= 9.0;
            }
            r.commit_changes();
        }
    }

    if use_graphics {
        std::thread::sleep(Duration::from_millis(1000));
    }

    state.reward.end_t = state.road.t;
    state.reward.avg_vel = state.reward.dist_travelled / state.road.t;
    state.reward.calculate_timestep_metrics();

    (state.road.cost, state.reward)
}

fn road_set_for_scenario(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
    n: usize,
) -> RoadSet {
    if params.use_cfb {
        let (base_set, _selected_ids) = conditional_focused_branching(params, true_road, n);
        base_set
    } else {
        RoadSet::new_samples(true_road, rng, n)
    }
}

fn main() {
    arg_parameters::run_parallel_scenarios();
}
