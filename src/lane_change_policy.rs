use nalgebra::point;
use parry2d_f64::na::Point2;

use crate::{
    car::{PREFERRED_VEL_ESTIMATE_MIN, PRIUS_LENGTH},
    road::LANE_WIDTH,
    side_policies::{SidePolicy, SidePolicyTrait},
    Road,
};

const TRANSITION_DIST_MIN: f64 = 1.0 * PRIUS_LENGTH;
const TRANSITION_DIST_MAX: f64 = 100.0 * PRIUS_LENGTH;

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum LongitudinalPolicy {
    Maintain,
    Accelerate,
    Decelerate,
}

#[derive(Clone, PartialEq, PartialOrd)]
pub struct LaneChangePolicy {
    policy_id: u32,
    target_lane_i: Option<i32>,
    transition_time: f64,
    wait_for_clear: bool,
    long_policy: LongitudinalPolicy,
    start_vel: Option<f64>,
    waiting_done: bool,
}

impl std::fmt::Debug for LaneChangePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        let policy_str = format_f!("{s.long_policy:?}");
        write_f!(f, "lane {s.target_lane_i:?}, {policy_str:10}")
    }
}

impl LaneChangePolicy {
    pub fn new(
        policy_id: u32,
        target_lane_i: Option<i32>,
        transition_time: f64,
        wait_for_clear: bool,
        long_policy: LongitudinalPolicy,
    ) -> Self {
        Self {
            policy_id,
            target_lane_i,
            transition_time,
            wait_for_clear,
            long_policy,
            start_vel: None,
            waiting_done: false,
        }
    }

    fn lane_change_trajectory(&mut self, road: &Road, car_i: usize, traj: &mut Vec<Point2<f64>>) {
        let car = &road.cars[car_i];

        let total_transition_dist = (self.transition_time * car.vel)
            .max(TRANSITION_DIST_MIN)
            .min(TRANSITION_DIST_MAX);

        let target_y = Road::get_lane_y(self.target_lane_i.unwrap_or_else(|| car.current_lane()));

        let transition_left = (car.y() - target_y).abs() / LANE_WIDTH;
        let transition_dist = total_transition_dist * transition_left;

        let target_x = car.x() + transition_dist;
        // let progress = (road.t - start_time) / self.transition_time;

        traj.clear();
        traj.extend_from_slice(&[
            point!(car.x(), car.y()),
            point!(target_x, target_y),
            point!(target_x + 100.0, target_y), // then continue straight
        ]);
    }

    fn lane_keep_trajectory(&mut self, road: &Road, car_i: usize, traj: &mut Vec<Point2<f64>>) {
        let car = &road.cars[car_i];
        let lane_i = car.current_lane();

        let transition_dist = (self.transition_time * car.vel)
            .max(TRANSITION_DIST_MIN)
            .min(TRANSITION_DIST_MAX);

        traj.clear();
        traj.extend_from_slice(&[
            point!(car.x(), car.y()),
            point!(car.x() + transition_dist, Road::get_lane_y(lane_i)),
            point!(car.x() + 100.0, Road::get_lane_y(lane_i)),
        ]);
    }
}

impl SidePolicyTrait for LaneChangePolicy {
    fn choose_target_lane(&mut self, road: &Road, car_i: usize) -> i32 {
        if self.wait_for_clear && !self.waiting_done {
            return road.cars[car_i].current_lane();
        }
        self.target_lane_i
            .unwrap_or_else(|| road.cars[car_i].current_lane())
    }

    fn choose_follow_time(&mut self, _road: &Road, _car_i: usize) -> f64 {
        match self.long_policy {
            LongitudinalPolicy::Maintain => 0.6,
            LongitudinalPolicy::Accelerate => 0.2,
            LongitudinalPolicy::Decelerate => 1.0,
        }
    }

    fn choose_vel(&mut self, road: &Road, car_i: usize) -> f64 {
        let car = &road.cars[car_i];
        let target_vel = match self.long_policy {
            LongitudinalPolicy::Maintain => self
                .start_vel
                .get_or_insert(car.vel)
                .max(PREFERRED_VEL_ESTIMATE_MIN),
            LongitudinalPolicy::Accelerate => (car.vel + 10.0).max(PREFERRED_VEL_ESTIMATE_MIN),
            LongitudinalPolicy::Decelerate => (car.vel - 10.0).max(0.0),
        };

        target_vel
    }

    fn choose_trajectory(&mut self, road: &Road, car_i: usize, traj: &mut Vec<Point2<f64>>) {
        if self.wait_for_clear && !self.waiting_done {
            let car = &road.cars[car_i];
            self.waiting_done = road.lane_definitely_clear_between(
                car_i,
                self.target_lane_i.unwrap_or_else(|| car.current_lane()),
                car.x() - 0.5 * car.length - car.length,
                car.x() + 0.5 * car.length,
            );
        }
        if self.waiting_done || !self.wait_for_clear {
            self.lane_change_trajectory(road, car_i, traj)
        } else {
            self.lane_keep_trajectory(road, car_i, traj)
        }
    }

    fn policy_id(&self) -> u32 {
        self.policy_id
        // use std::hash::{Hash, Hasher};
        // let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // self.target_lane_i.hash(&mut hasher);
        // self.transition_time.to_bits().hash(&mut hasher);
        // if let Some(follow_time) = self.follow_time {
        //     follow_time.to_bits().hash(&mut hasher);
        // }
        // hasher.finish()
    }

    fn operating_policy(&self) -> SidePolicy {
        SidePolicy::LaneChangePolicy(self.clone())
    }
}
