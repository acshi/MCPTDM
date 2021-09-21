use itertools::Itertools;
use rand::{
    distributions::WeightedIndex,
    prelude::{Distribution, StdRng},
};

use crate::{lane_change_policy::LongitudinalPolicy, road::Road};

fn predict_lane(road: &Road, car_i: usize) -> i32 {
    let car = &road.cars[car_i];
    let predicted_y =
        car.y() + car.vel * (car.theta() + car.steer).sin() * road.params.lane_change_time;
    Road::get_lane_i(predicted_y).min(1).max(0)
}

fn predict_long(road: &Road, car_i: usize) -> LongitudinalPolicy {
    let lane_i = road.cars[car_i].current_lane();
    let ahead_dist = road.dist_clear_ahead_in_lane(car_i, lane_i);
    let bparams = &road.params.belief;
    let car = &road.cars[car_i];
    if let Some((ahead_dist, ahead_car_i)) = ahead_dist {
        let ahead_car = &road.cars[ahead_car_i];
        if car.vel > ahead_car.vel + bparams.accelerate_delta_vel_thresh
            || ahead_dist < bparams.accelerate_ahead_dist_thresh
        {
            return LongitudinalPolicy::Accelerate;
        } else {
            return LongitudinalPolicy::Maintain;
        }
    }
    if car.vel < bparams.decelerate_vel_thresh {
        LongitudinalPolicy::Decelerate
    } else {
        LongitudinalPolicy::Accelerate
    }
}

fn predict_finished_waiting(road: &Road, car_i: usize) -> bool {
    let car = &road.cars[car_i];
    let lane_y = Road::get_lane_y(car.current_lane());
    let dy = (lane_y - car.y()).abs();
    dy > road.params.belief.finished_waiting_dy
}

fn normalize(belief: &mut [f64]) {
    let sum: f64 = belief.iter().sum();
    for val in belief.iter_mut() {
        *val /= sum;
    }
}

#[derive(Clone)]
pub struct Belief {
    belief: Vec<Vec<f64>>,
}
impl Belief {
    pub fn uniform(n_cars: usize, n_policies: usize) -> Self {
        Self {
            belief: vec![vec![1.0 / n_policies as f64; n_policies]; n_cars],
        }
    }

    #[allow(unused)]
    pub fn for_all_cars(n_cars: usize, belief: &[f64]) -> Self {
        let mut single_belief = belief.to_vec();
        normalize(&mut single_belief);

        Self {
            belief: vec![single_belief; n_cars],
        }
    }

    pub fn update(&mut self, road: &Road) {
        let bparams = &road.params.belief;
        for (car_i, belief) in self.belief.iter_mut().enumerate().skip(1) {
            let pred_lane = predict_lane(road, car_i);
            let pred_long = predict_long(road, car_i);
            let pred_finished_waiting = predict_finished_waiting(road, car_i);

            if road.super_debug()
                && road.params.belief_debug
                && road.params.debug_car_i == Some(car_i)
            {
                eprintln_f!("{pred_lane=} {pred_long=:?} {pred_finished_waiting=}");
            }

            belief.clear();
            for &lane_i in &[0, 1] {
                for long_policy in [LongitudinalPolicy::Maintain, LongitudinalPolicy::Accelerate] {
                    for wait_for_clear in [false, true] {
                        let mut prob = 1.0;
                        if lane_i != pred_lane {
                            prob *= bparams.different_lane_prob;
                        }
                        if long_policy != pred_long {
                            prob *= bparams.different_longitudinal_prob;
                        }
                        // wait_for_clear && pred_finished_waiting: already making lane change
                        // !wait_for_clear && pred_finished_waiting: already making lane change
                        // wait_for_clear && !pred_finished_waiting: still need to wait
                        // !wait_for_clear && !pred_finished_waiting: will start lane change
                        let would_lane_change = pred_finished_waiting || !wait_for_clear;
                        let current_lane_i = road.cars[car_i].current_lane();
                        let wants_lane_change = lane_i != current_lane_i;
                        let will_lane_change = would_lane_change && wants_lane_change;
                        // either we can make the lane change, and might as well use wait_for_clear=false
                        // or we still need to wait and so use wait_for_clear=true
                        // the other scenarios are superfluous, or inaccurate
                        if will_lane_change && wait_for_clear {
                            prob = 0.0;
                        }
                        // waiting... to _not_ change lanes is also pointless
                        if !wants_lane_change && wait_for_clear {
                            prob = 0.0;
                        }
                        // the chance that the vehicle effectively skips checking for it to be clear before turning
                        // in practice, this would more mean that noise prevented us from telling that they already started turning(?)
                        if wants_lane_change && !pred_finished_waiting && !wait_for_clear {
                            prob *= bparams.skips_waiting_prob;
                        }
                        belief.push(prob);

                        if road.super_debug()
                            && road.params.belief_debug
                            && road.params.debug_car_i == Some(car_i)
                        {
                            eprintln_f!("{road.timesteps}: {car_i=} {lane_i=} {long_policy=:?} {wait_for_clear=}: {prob=:.2}, would: {would_lane_change}, wants: {wants_lane_change}, will: {will_lane_change}");
                        }
                    }
                }
            }
            if LongitudinalPolicy::Decelerate == pred_long {
                belief.push(bparams.decelerate_prior_prob);
            } else {
                belief.push(bparams.decelerate_prior_prob * bparams.different_longitudinal_prob);
            }

            normalize(belief);

            if road.params.belief_debug
                && road.super_debug()
                && road.params.debug_car_i == Some(car_i)
            {
                eprintln_f!("{road.timesteps}: Belief about {car_i}: {belief:.2?}");
            }
        }
    }

    pub fn sample(&self, rng: &mut StdRng) -> Vec<usize> {
        self.belief
            .iter()
            .map(|weights| WeightedIndex::new(weights).unwrap().sample(rng))
            .collect_vec()
    }

    pub fn get(&self, car_i: usize, policy_id: usize) -> f64 {
        assert_ne!(car_i, 0);
        self.belief[car_i][policy_id]
    }

    pub fn get_all(&self, car_i: usize) -> &[f64] {
        &self.belief[car_i]
    }

    pub fn get_most_likely(&self, car_i: usize) -> usize {
        assert_ne!(car_i, 0);
        self.belief[car_i]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    pub fn is_uncertain(&self, car_i: usize, threshold: f64) -> bool {
        assert_ne!(car_i, 0);
        if self.belief[car_i].len() <= 1 {
            return false;
        }

        let mut values = self.belief[car_i].clone();

        // sort descending (switched a and b)
        values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        (values[0] - values[1]) < threshold
    }
}
