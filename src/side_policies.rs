use parry2d_f64::na::Point2;

use crate::delayed_policy::DelayedPolicy;
use crate::lane_change_policy::LaneChangePolicy;
use crate::open_loop_policy::OpenLoopPolicy;
use crate::Road;

#[enum_dispatch]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SidePolicy {
    LaneChangePolicy,
    DelayedPolicy,
    OpenLoopPolicy,
}

#[enum_dispatch(SidePolicy)]
pub trait SidePolicyTrait {
    fn precheck(&mut self, _road: &Road, _dt: f64) {}

    fn choose_target_lane(&mut self, road: &Road, car_i: usize) -> i32 {
        road.cars[car_i].target_lane_i
    }

    fn choose_follow_time(&mut self, road: &crate::Road, car_i: usize) -> f64 {
        road.cars[car_i].preferred_follow_time
    }

    fn choose_vel(&mut self, road: &Road, car_i: usize) -> f64 {
        road.cars[car_i].preferred_vel
    }

    fn choose_trajectory(&mut self, road: &Road, car_i: usize, traj: &mut Vec<Point2<f64>>);
    fn policy_id(&self) -> u32;
    fn operating_policy(&self) -> SidePolicy;
}
