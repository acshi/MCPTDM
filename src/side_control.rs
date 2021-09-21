use parry2d_f64::na::Point2;
use rvx::Rvx;

use crate::Road;

use crate::open_loop_policy::OpenLoopSideControl;
use crate::pure_pursuit::PurePursuitPolicy;

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum SideControl {
    PurePursuitPolicy,
    OpenLoopSideControl,
}

#[enum_dispatch(SideControl)]
pub trait SideControlTrait {
    fn choose_steer(&mut self, road: &Road, car_i: usize, trajectory: &[Point2<f64>]) -> f64;

    fn draw(&self, r: &mut Rvx) {
        let _ = r;
    }
}
