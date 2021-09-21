use crate::{car::BREAKING_ACCEL, forward_control::ForwardControlTrait, Road};

#[derive(Debug, Clone)]
pub struct IntelligentDriverPolicy;

impl IntelligentDriverPolicy {
    pub fn new() -> Self {
        Self
    }
}

// https://en.wikipedia.org/wiki/Intelligent_driver_model
impl ForwardControlTrait for IntelligentDriverPolicy {
    fn choose_accel(&mut self, road: &Road, car_i: usize) -> f64 {
        let car = &road.cars[car_i];

        let accel_free_road = if car.target_vel == 0.0 {
            if car.vel > 0.0 {
                -BREAKING_ACCEL
            } else {
                0.0
            }
        } else {
            car.preferred_accel * (1.0 - (car.vel / car.target_vel).powi(4))
        };

        // if road.params.intelligent_driver_debug && road.super_debug() && car.is_ego() {
        //     eprintln_f!(
        //         "{road.timesteps}: {car.vel=:.4} {car.preferred_accel=:.4}, {car.target_vel=:.4}"
        //     );
        // }

        assert!(
            accel_free_road.is_finite(),
            "Bad accel_free_road: {}, w/ vel {:.2}, target_vel: {:.2}",
            accel_free_road,
            car.vel,
            car.target_vel,
        );

        let accel;
        if let Some((forward_dist, c_i)) = road.dist_clear_ahead_in_lane(car_i, car.target_lane_i) {
            let approaching_rate = car.vel - road.cars[c_i].vel;

            let follow_dist = car.follow_dist();
            let spacing_term = follow_dist
                + car.vel * approaching_rate
                    / (2.0 * (car.preferred_accel * BREAKING_ACCEL).sqrt());
            let accel_interaction = car.preferred_accel * (-(spacing_term / forward_dist).powi(2));

            accel = accel_free_road + accel_interaction;

            if road.params.intelligent_driver_debug {
                if road.super_debug() && car.is_ego() {
                    eprintln_f!("{road.timesteps}: {car_i=}, {c_i=}, lane_i = {car.target_lane_i}, {forward_dist=:.10}, {follow_dist=:.10}, vel = {car.vel:.10}, {approaching_rate=:.10}, {spacing_term=:.10}, {accel_free_road=:.10}, {accel_interaction=:.10}");
                } else if road.super_debug() && c_i == 0 && road.params.debug_car_i == Some(car_i) {
                    eprintln_f!("{road.timesteps}: {car_i=}, {c_i=}, lane_i = {car.target_lane_i}, {forward_dist=:.10}, {follow_dist=:.10}, vel = {car.vel:.10}, {approaching_rate=:.10}, {spacing_term=:.10}, {accel_free_road=:.10}, {accel_interaction=:.10}");
                }
            }
        } else {
            accel = accel_free_road;

            if road.params.intelligent_driver_debug && road.super_debug() && car.is_ego() {
                eprintln_f!(
                    "{road.timesteps}: {car_i=}, lane_i = {car.target_lane_i}, vel = {car.vel:.10}, {accel_free_road=:6.10}, {car.target_vel=:.10}"
                );
            }
        }

        accel
    }
}
