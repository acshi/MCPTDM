use std::f64::consts::PI;

use nalgebra::vector;
use parry2d_f64::{
    bounding_volume::AABB,
    na::Isometry2,
    shape::{Cuboid, Shape},
};
use rand::prelude::{Rng, StdRng};
use rvx::{Rvx, RvxColor};

use crate::{
    arg_parameters::Parameters,
    forward_control::ForwardControl,
    intelligent_driver::IntelligentDriverPolicy,
    mpdm::make_obstacle_vehicle_policy_choices,
    open_loop_policy::{OpenLoopForwardControl, OpenLoopPolicy, OpenLoopSideControl},
    pure_pursuit::PurePursuitPolicy,
    road::{Road, ROAD_LENGTH},
    side_control::{SideControl, SideControlTrait},
    side_policies::{SidePolicy, SidePolicyTrait},
    AHEAD_TIME_DEFAULT,
};

pub const PRIUS_WIDTH: f64 = 1.76;
pub const PRIUS_LENGTH: f64 = 4.57;
pub const PRIUS_MAX_STEER: f64 = 1.11; // from minimum turning radius of 4.34 meters and PRIUS_LENGTH
pub const MPH_TO_MPS: f64 = 0.44704;
pub const MPS_TO_MPH: f64 = 2.23694;
pub const SPEED_DEFAULT: f64 = 25.0 * MPH_TO_MPS;
pub const SPEED_LOW: f64 = 15.0 * MPH_TO_MPS;
pub const SPEED_HIGH: f64 = 35.0 * MPH_TO_MPS;
pub const FOLLOW_DIST_BASE: f64 = 1.5 * PRIUS_LENGTH;
pub const FOLLOW_TIME_LOW: f64 = 0.8;
pub const FOLLOW_TIME_HIGH: f64 = 2.0;
pub const FOLLOW_TIME_DEFAULT: f64 = 1.2;

pub const PREFERRED_VEL_ESTIMATE_MIN: f64 = 5.0 * MPH_TO_MPS;

pub const PREFERRED_ACCEL_LOW: f64 = 1.0; // 0.2; // semi truck, 2min zero to sixty
pub const PREFERRED_ACCEL_HIGH: f64 = 2.0; // 11.2; // model s, 2.4s zero to sixty
pub const PREFERRED_ACCEL_DEFAULT: f64 = 2.0; // 16s zero to sixty, just under max accel for a prius (13s)
pub const BREAKING_ACCEL: f64 = 6.0;

#[derive(Clone, Debug)]
pub struct Car {
    pub car_i: usize,
    pub crashed: bool,

    // front-referenced kinematic bicycle model
    x: f64,
    y: f64,
    theta: f64,
    pub vel: f64,
    pub steer: f64,

    pub width: f64,
    pub length: f64,

    // "attitude" properties/constants
    pub preferred_vel: f64,
    pub preferred_accel: f64,
    pub preferred_follow_time: f64,

    // current properties/goals
    pub target_follow_time: f64,
    pub target_vel: f64,
    pub target_lane_i: i32,

    pub forward_control: Option<ForwardControl>,
    pub side_control: Option<SideControl>,
    pub side_policy: Option<SidePolicy>,

    // cached
    shape: Cuboid,
    pose: Isometry2<f64>,
    aabb: AABB,
    // norotation_aabb: AABB,
}

impl Car {
    pub fn new(params: &Parameters, car_i: usize, lane_i: i32) -> Self {
        let lane_y = Road::get_lane_y(lane_i);
        let policies = make_obstacle_vehicle_policy_choices(params);
        let width = PRIUS_WIDTH;
        let length = PRIUS_LENGTH;
        let mut car = Self {
            car_i,
            crashed: false,

            x: 0.0,
            y: lane_y,
            theta: 0.0,
            vel: 0.0,
            steer: 0.0,

            width,
            length,

            preferred_vel: SPEED_DEFAULT,
            preferred_accel: PREFERRED_ACCEL_DEFAULT,
            preferred_follow_time: FOLLOW_TIME_DEFAULT,

            target_follow_time: FOLLOW_TIME_DEFAULT,
            target_vel: SPEED_DEFAULT,
            target_lane_i: lane_i,

            // policy: Some(Policy::AdapativeCruisePolicy(AdapativeCruisePolicy::new())),
            forward_control: Some(ForwardControl::IntelligentDriverPolicy(
                IntelligentDriverPolicy::new(),
            )),
            side_control: Some(SideControl::PurePursuitPolicy(PurePursuitPolicy::new(
                AHEAD_TIME_DEFAULT,
            ))),
            side_policy: Some(if lane_i == 0 {
                policies[1].clone()
            } else {
                policies[3].clone()
            }),

            shape: Cuboid::new(vector!(length / 2.0, width / 2.0)),
            pose: Isometry2::identity(),
            aabb: AABB::new_invalid(),
        };

        car.update_geometry_cache();
        car
    }

    pub fn random_new(params: &Parameters, car_i: usize, rng: &mut StdRng) -> Self {
        let lane_i = rng.gen_range(0..=1);
        let mut car = Self::new(params, car_i, lane_i);
        car.preferred_vel = rng.gen_range(SPEED_LOW..SPEED_HIGH);
        car.vel = car.preferred_vel;
        car.set_x(rng.gen_range(0.0..ROAD_LENGTH) - ROAD_LENGTH / 2.0);
        car.preferred_accel = rng.gen_range(PREFERRED_ACCEL_LOW..PREFERRED_ACCEL_HIGH);
        car.preferred_follow_time = rng.gen_range(FOLLOW_TIME_LOW..FOLLOW_TIME_HIGH);

        car
    }

    pub fn sim_estimate(&self) -> Self {
        let mut sim_car = self.clone();

        sim_car.preferred_vel = self.vel.max(SPEED_LOW);
        sim_car.preferred_accel = PREFERRED_ACCEL_DEFAULT;
        sim_car.preferred_follow_time = FOLLOW_TIME_DEFAULT;

        sim_car.target_lane_i = sim_car.current_lane();
        sim_car.target_vel = sim_car.vel;
        sim_car.target_follow_time = sim_car.preferred_follow_time;

        sim_car
    }

    pub fn open_loop_estimate(&self) -> Self {
        let mut car = self.sim_estimate();

        car.side_policy = Some(SidePolicy::OpenLoopPolicy(OpenLoopPolicy));
        car.side_control = Some(SideControl::OpenLoopSideControl(OpenLoopSideControl));
        car.forward_control = Some(ForwardControl::OpenLoopForwardControl(
            OpenLoopForwardControl,
        ));

        car
    }

    pub fn operating_policy_id(&self) -> u32 {
        self.side_policy
            .as_ref()
            .unwrap()
            .operating_policy()
            .policy_id()
    }

    pub fn full_policy_id(&self) -> u32 {
        self.side_policy.as_ref().unwrap().policy_id()
    }

    pub fn is_ego(&self) -> bool {
        self.car_i == 0
    }

    pub fn follow_dist(&self) -> f64 {
        FOLLOW_DIST_BASE + self.target_follow_time * self.vel
    }

    fn update_geometry_cache(&mut self) {
        let center_x = self.x - self.length / 2.0 * self.theta.cos();
        let center_y = self.y - self.length / 2.0 * self.theta.sin();
        self.pose = Isometry2::new(vector!(center_x, center_y), self.theta);

        self.aabb = self.shape().compute_aabb(&self.pose());
    }

    pub fn update(&mut self, dt: f64) {
        if !self.crashed {
            let theta = self.theta + self.steer;
            self.x += theta.cos() * self.vel * dt;
            self.y += theta.sin() * self.vel * dt;
            self.theta += self.vel * self.steer.sin() / self.length * dt;

            self.update_geometry_cache();
        }
    }

    pub fn draw(&self, params: &Parameters, r: &mut Rvx, color: RvxColor) {
        // front dot
        r.draw(
            Rvx::circle()
                .scale(0.5)
                .translate(&[self.x, self.y])
                .color(RvxColor::WHITE.set_a(0.5)),
        );

        // back dot
        r.draw(
            Rvx::circle()
                .scale(0.5)
                .translate(&[
                    self.x - self.length * self.theta.cos(),
                    self.y - self.length * self.theta.sin(),
                ])
                .color(RvxColor::YELLOW.set_a(0.5)),
        );

        // front wheel
        r.draw(
            Rvx::square()
                .scale_xy(&[1.0, 0.5])
                .rot(self.theta + self.steer)
                .translate(&[self.x, self.y])
                .color(RvxColor::BLACK.set_a(0.9)),
        );

        // back wheel
        r.draw(
            Rvx::square()
                .scale_xy(&[1.0, 0.5])
                .rot(self.theta)
                .translate(&[
                    self.x - self.length * self.theta.cos(),
                    self.y - self.length * self.theta.sin(),
                ])
                .color(RvxColor::BLACK.set_a(0.9)),
        );

        let center_x = self.x - self.length / 2.0 * self.theta.cos();
        let center_y = self.y - self.length / 2.0 * self.theta.sin();

        r.draw(
            Rvx::square()
                .scale_xy(&[self.length, self.width])
                .rot(self.theta)
                .translate(&[center_x, center_y])
                .color(color),
        );

        // if !params.graphics_for_paper {
        r.draw(
            Rvx::text(&format!("{:.1}", self.car_i,), "Arial", 60.0)
                .rot(-PI / 2.0)
                .translate(&[self.x - self.length / 2.0, self.y + self.width / 2.0])
                .color(RvxColor::BLACK),
        );
        // }

        if false {
            r.draw(
            Rvx::text(
                &format!(
                    "MPH: {:.1}\nPref MPH: {:.1}\nLane: {}, y: {:.2}\nFollow time: {:.1}\nPref accel: {:.1}\nPref follow time: {:.1}\nPref follow: {:.1}",
                    self.vel * MPS_TO_MPH,
                    self.preferred_vel * MPS_TO_MPH,
                    self.current_lane(),
                    self.y,
                    self.target_follow_time,
                    self.preferred_accel,
                    self.preferred_follow_time,
                    self.follow_dist(),
                ),
                "Arial",
                40.0,
            )
            .rot(-PI / 2.0)
            .translate(&[self.x, self.y]),
        );
        } else if !params.graphics_for_paper {
            r.draw(
                Rvx::text(
                    &format!(
                        "MPH: {:.1}\nPref MPH: {:.1}\nFollow time: {:.1}\nx: {:.1}\n{}",
                        self.vel * MPS_TO_MPH,
                        self.preferred_vel * MPS_TO_MPH,
                        self.target_follow_time,
                        self.x,
                        if self.is_ego() || params.debug_car_i == Some(self.car_i) {
                            format!("{:?}", self.side_policy.as_ref().unwrap())
                        } else {
                            "".to_owned()
                        },
                    ),
                    "Arial",
                    40.0,
                )
                .rot(-PI / 2.0)
                .translate(&[self.x, self.y]),
            );
        }

        if self.is_ego() && !params.graphics_for_paper {
            self.side_control.iter().for_each(|a| a.draw(r));
        }
    }

    pub fn current_lane(&self) -> i32 {
        Road::get_lane_i(self.y)
    }

    pub fn shape(&self) -> impl Shape {
        self.shape
    }

    pub fn pose(&self) -> Isometry2<f64> {
        // let center_x = self.x - self.length / 2.0 * self.theta.cos();
        // let center_y = self.y - self.length / 2.0 * self.theta.sin();
        // let pose = Isometry2::new(vector!(center_x, center_y), self.theta);
        // assert_eq!(pose, self.pose);

        self.pose
    }

    pub fn aabb(&self) -> AABB {
        // let aabb = self.shape().compute_aabb(&self.pose());
        // assert_eq!(aabb, self.aabb);

        self.aabb
    }

    pub fn x(&self) -> f64 {
        self.x
    }

    pub fn y(&self) -> f64 {
        self.y
    }

    pub fn theta(&self) -> f64 {
        self.theta
    }

    pub fn set_x(&mut self, x: f64) {
        self.x = x;
        self.update_geometry_cache();
    }

    #[allow(unused)]
    pub fn set_y(&mut self, y: f64) {
        self.y = y;
        self.update_geometry_cache();
    }

    #[allow(unused)]
    pub fn set_theta(&mut self, theta: f64) {
        self.theta = theta;
        self.update_geometry_cache();
    }

    pub fn spatial_x(&self) -> i32 {
        self.spatial_offset(0.0)
    }

    pub fn spatial_offset(&self, dx: f64) -> i32 {
        ((self.x + dx) * 1000.0) as i32
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SpatialCar {
    pub x: i32,
    pub car_i: u32,
}

impl From<&Car> for SpatialCar {
    fn from(item: &Car) -> Self {
        Self {
            x: item.spatial_x(),
            car_i: item.car_i as u32,
        }
    }
}
