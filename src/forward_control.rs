use crate::intelligent_driver::IntelligentDriverPolicy;
use crate::open_loop_policy::OpenLoopForwardControl;
use crate::Road;

#[enum_dispatch]
#[derive(Debug, Clone)]
pub enum ForwardControl {
    IntelligentDriverPolicy,
    OpenLoopForwardControl,
}

#[enum_dispatch(ForwardControl)]
pub trait ForwardControlTrait {
    fn choose_accel(&mut self, road: &Road, car_i: usize) -> f64;
}
