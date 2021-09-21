use parry2d_f64::na::Point2;

use crate::{
    road::Road,
    side_policies::{SidePolicy, SidePolicyTrait},
};

#[derive(Clone, PartialEq, PartialOrd)]
pub struct DelayedPolicy {
    policy_a: Box<SidePolicy>,
    policy_b: Box<SidePolicy>,
    delay_time: f64,
    start_time: Option<f64>,
    time_until_switch: f64,
    has_switched: bool,
}

impl std::fmt::Debug for DelayedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        writeln!(f, "policy_id: {:?}", self.policy_id())?;
        writeln!(f, "policy_a: {:?}", self.policy_a)?;
        writeln!(f, "policy_b: {:?}", self.policy_b)?;
        writeln!(f, "delay_time: {:.2?}", self.delay_time)?;
        writeln!(f, "start_time: {:.2?}", self.start_time)?;
        writeln!(f, "time_until_switch: {:.2?}", self.time_until_switch)?;
        writeln!(f, "has_switched: {:?}", self.has_switched)
    }
}

impl DelayedPolicy {
    pub fn new(policy_a: SidePolicy, policy_b: SidePolicy, delay_time: f64) -> Self {
        Self {
            policy_a: Box::new(policy_a),
            policy_b: Box::new(policy_b),
            delay_time,
            start_time: None,
            time_until_switch: delay_time,
            has_switched: false,
        }
    }

    fn check_for_switch(&mut self, road: &Road, dt: f64) {
        let start_time = *self.start_time.get_or_insert(road.t);
        self.time_until_switch = (start_time + self.delay_time - road.t).max(0.0);
        if self.time_until_switch < dt {
            self.has_switched = true;
        }
    }
}

impl SidePolicyTrait for DelayedPolicy {
    fn precheck(&mut self, road: &Road, dt: f64) {
        self.check_for_switch(road, dt);
    }

    fn choose_target_lane(&mut self, road: &Road, car_i: usize) -> i32 {
        if self.has_switched {
            self.policy_b.choose_target_lane(road, car_i)
        } else {
            self.policy_a.choose_target_lane(road, car_i)
        }
    }

    fn choose_trajectory(&mut self, road: &Road, car_i: usize, traj: &mut Vec<Point2<f64>>) {
        if self.has_switched {
            self.policy_b.choose_trajectory(road, car_i, traj)
        } else {
            self.policy_a.choose_trajectory(road, car_i, traj)
        }
    }

    fn choose_follow_time(&mut self, road: &crate::Road, car_i: usize) -> f64 {
        if self.has_switched {
            self.policy_b.choose_follow_time(road, car_i)
        } else {
            self.policy_a.choose_follow_time(road, car_i)
        }
    }

    fn choose_vel(&mut self, road: &Road, car_i: usize) -> f64 {
        if self.has_switched {
            self.policy_b.choose_vel(road, car_i)
        } else {
            self.policy_a.choose_vel(road, car_i)
        }
    }

    fn policy_id(&self) -> u32 {
        let mut policy_id = 100 + self.delay_time as u32 * 1000;
        if self.has_switched {
            policy_id += 10 * self.policy_b.policy_id();
        } else {
            policy_id += 10 * self.policy_a.policy_id();
        }
        policy_id += self.policy_b.policy_id();
        policy_id

        // use std::hash::{Hash, Hasher};
        // let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // if self.has_switched {
        //     self.policy_b.policy_id().hash(&mut hasher);
        // } else {
        //     self.policy_a.policy_id().hash(&mut hasher);
        // }
        // self.policy_b.policy_id().hash(&mut hasher);
        // self.delay_time.to_bits().hash(&mut hasher);
        // hasher.finish()
    }

    fn operating_policy(&self) -> SidePolicy {
        if self.has_switched {
            self.policy_b.operating_policy()
        } else {
            self.policy_a.operating_policy()
        }
    }
}
