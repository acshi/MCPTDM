use crate::{
    forward_control::ForwardControlTrait,
    side_control::SideControlTrait,
    side_policies::{SidePolicy, SidePolicyTrait},
};

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct OpenLoopPolicy;

impl SidePolicyTrait for OpenLoopPolicy {
    fn choose_trajectory(
        &mut self,
        _road: &crate::road::Road,
        _car_i: usize,
        _traj: &mut Vec<nalgebra::Point2<f64>>,
    ) {
    }

    fn policy_id(&self) -> u32 {
        1000
    }

    fn operating_policy(&self) -> SidePolicy {
        SidePolicy::OpenLoopPolicy(self.clone())
    }
}

#[derive(Clone, Debug)]
pub struct OpenLoopSideControl;

impl SideControlTrait for OpenLoopSideControl {
    fn choose_steer(
        &mut self,
        _road: &crate::road::Road,
        _car_i: usize,
        _trajectory: &[nalgebra::Point2<f64>],
    ) -> f64 {
        0.0
    }
}

#[derive(Clone, Debug)]
pub struct OpenLoopForwardControl;

impl ForwardControlTrait for OpenLoopForwardControl {
    fn choose_accel(&mut self, _road: &crate::road::Road, _car_i: usize) -> f64 {
        0.0
    }
}
