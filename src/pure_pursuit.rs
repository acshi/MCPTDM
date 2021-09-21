use nalgebra::point;
use parry2d_f64::{math::Isometry, na::Point2, shape::Ball};
use rvx::{Rvx, RvxColor};

use crate::{car::PRIUS_LENGTH, road::LANE_WIDTH, side_control::SideControlTrait, Road};
use itertools::Itertools;

const AHEAD_DIST_MIN: f64 = LANE_WIDTH + PRIUS_LENGTH * 0.2;
const AHEAD_DIST_MAX: f64 = 20.0 * PRIUS_LENGTH;

#[derive(Clone)]
struct PurePursuitPolicyDebug {
    target_x: f64,
    target_y: f64,
    car_x: f64,
    car_y: f64,
    ahead_dist: f64,
    trajectory: Vec<Point2<f64>>,
}

impl PurePursuitPolicyDebug {
    fn new() -> Self {
        Self {
            target_x: 0.0,
            target_y: 0.0,
            trajectory: vec![Point2::new(0.0, 0.0)],
            car_x: 0.0,
            car_y: 0.0,
            ahead_dist: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct PurePursuitPolicy {
    ahead_time: f64,
    debug_info: Option<Box<PurePursuitPolicyDebug>>,
}

impl std::fmt::Debug for PurePursuitPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PurePursuitPolicy")
    }
}

impl PurePursuitPolicy {
    pub fn new(ahead_time: f64) -> Self {
        Self {
            ahead_time,
            debug_info: None,
        }
    }
}

fn ranges_overlap(low_a: f64, high_a: f64, low_b: f64, high_b: f64) -> bool {
    if low_a - high_b <= 0.0 {
        return true;
    }
    if low_b - high_a <= 0.0 {
        return true;
    }
    false
}

fn circle_line_contact(
    circ_xy: Point2<f64>,
    radius: f64,
    pt1: Point2<f64>,
    pt2: Point2<f64>,
) -> Option<Point2<f64>> {
    // the intersection of a line (in two-point form) and a circle
    // forms a quadratic equation.
    // If there are solutions, then we have an intersection

    // Weisstein, Eric W. "Circle-Line Intersection." From MathWorld--A Wolfram Web Resource.
    // https://mathworld.wolfram.com/Circle-LineIntersection.html
    let (cx, cy) = (circ_xy.x, circ_xy.y);
    let r = radius;

    let (x1, y1) = (pt1.x - cx, pt1.y - cy);
    let (x2, y2) = (pt2.x - cx, pt2.y - cy);
    // eprint_f!("{cx=:6.2} {cy=:6.2}, {x1=:6.2} {y1=:6.2}, {x2=:6.2} {y2=:6.2}, {r:6.2}: ");

    // handle some special cases a little faster
    if y1 == y2 {
        if y1.abs() > radius {
            return None;
        }
        let r_prime = if y1 == 0.0 {
            r
        } else {
            (r * r - y1 * y1).sqrt()
        };
        let (low_x, high_x) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
        if ranges_overlap(low_x, high_x, -r_prime, r_prime) {
            if x1 < 0.0 {
                return Some(point!(cx - r_prime, y1 + cy));
            } else {
                return Some(point!(cx + r_prime, y1 + cy));
            }
        }
        return None;
    }

    let dx = x2 - x1;
    let dy = y2 - y1;
    let dxy = x1 * y2 - x2 * y1;
    let dr2 = dx * dx + dy * dy;
    let disc = r * r * dr2 - dxy * dxy;
    // eprintln_f!("{dx=:6.2} {dy=:6.2} {disc=:6.2}");

    // we have only shown that there is no collision with the _infinite_ line!
    if disc < 0.0 {
        return None;
    }

    // the above check does not work for y1 == y2, nor does the below extra check!
    // special case for when the y values are equal
    // because then the following logic will be invalid
    if y1 == y2 && y1 >= -r && y1 <= r {
        // easy! We can just check the x values. reminder that 0 is the circle center.
        // we also generally want to keep the solution closer to pt1/x1
        // first compute the effective radius at this y-level
        let efr = (r * r - y1 * y1).sqrt();
        // eprintln_f!("{efr=:.2}");
        if x1 > -efr && x1 < efr {
            // x1 is contained, so x2 will indicate which side of the circle is crossed
            // or whether the line is completely contained
            if x2 > -efr && x2 < efr {
                // completely contained
                return None;
                // return Some(pt1);
            }
            if x2 >= efr {
                return Some(Point2::new(efr + cx, pt1.y));
            }
            return Some(Point2::new(-efr + cx, pt1.y));
        }

        if x2 > -efr && x2 < efr {
            // x2 is contained, so x1 will indicate which side of the circle is crossed
            // it can't be completely contained because we already checked above
            if x1 >= efr {
                return Some(Point2::new(efr + cx, pt1.y));
            }
            return Some(Point2::new(-efr + cx, pt1.y));
        }

        if x1 * x2 < 0.0 {
            // neither point is contained, but they go through the center of the circle
            // so we have two contact points, and choose the one closer to pt1
            if x1 <= 0.0 {
                return Some(Point2::new(-r + cx, pt1.y));
            }
            return Some(Point2::new(r + cx, pt1.y));
        }

        // if x1 * x2 < 0.0 || x1 >= -r && x1 <= r || x2 >= -r && x2 <= r {
        //     if x1 <= 0.0 {
        //         return Some(Point2::new(-r + cx, pt1.y));
        //     } else {
        //         return Some(Point2::new(r + cx, pt1.y));
        //     }
        // }
        return None;
    }

    // now we can solve for y and see if the intersection points are on the finite line
    let factor1 = -dxy * dx;
    let disc_sqrt = disc.sqrt();
    let factor2 = dy.abs() * disc_sqrt;
    let dr2_recip = 1.0 / dr2;
    let coll_y1 = (factor1 + factor2) * dr2_recip;
    let coll_y2 = (factor1 - factor2) * dr2_recip;

    let mut coll_y1_valid = coll_y1 >= y1 && coll_y1 <= y2 || coll_y1 <= y1 && coll_y1 >= y2;
    let mut coll_y2_valid = coll_y2 >= y1 && coll_y2 <= y2 || coll_y2 <= y1 && coll_y2 >= y2;
    // eprintln_f!("{coll_y1=:.2} {coll_y2=:.2} {coll_y1_valid=}, {coll_y2_valid=}");

    if coll_y1_valid && coll_y2_valid {
        // keep the one solution closer to pt1 than pt2
        if (y1 < y2) == (coll_y1 < coll_y2) {
            coll_y2_valid = false;
        } else {
            coll_y1_valid = false;
        }
    }

    if coll_y1_valid {
        // eprintln!("using coll_y1");
        // let coll_rel_x = (r.powi(2) - coll_y1.powi(2)).sqrt() * (x1 * coll_y1).signum();
        let s = dx / dy;
        let coll_rel_x = s * coll_y1 + x1 - s * y1;
        let coll_x = coll_rel_x + cx;
        let coll_y = coll_y1 + cy;
        let coll_pt = Point2::new(coll_x, coll_y);
        // if let Some(y_res) = y_equals_result {
        //     assert!(y_res.is_some())
        // }
        Some(coll_pt)
    } else if coll_y2_valid {
        // eprintln!("using coll_y2");
        // let coll_rel_x = (r.powi(2) - coll_y2.powi(2)).sqrt() * (x2 * coll_y2).signum();
        let s = dx / dy;
        let coll_rel_x = s * coll_y2 + x1 - s * y1;
        let coll_x = coll_rel_x + cx;
        let coll_y = coll_y2 + cy;
        let coll_pt = Point2::new(coll_x, coll_y);
        // if let Some(y_res) = y_equals_result {
        //     assert!(y_res.is_some())
        // }
        Some(coll_pt)
    } else {
        // if let Some(y_res) = y_equals_result {
        //     // assert!(y_res.is_none())
        //     // if y_res.is_some() {
        //     //     eprintln!("Do we really have a collision between {:.2?} -> {:.2?} and {:.2?}", pt1, pt2, self);
        //     //     eprintln!("x1: {:.2} x2: {:.2}", x1, x2);
        //     //     eprintln!("y1: {:.2} y2: {:.2}", y1, y2);
        //     //     panic!();
        //     // }
        // }
        None
    }
}

fn polyline_contact(
    m1: &Isometry<f64>,
    polyline: &[Point2<f64>],
    m2: &Isometry<f64>,
    g2: &Ball,
    _prediction: f64,
) -> Option<Point2<f64>> {
    // in reverse to favor choosing the contact farther forward
    for (a, b) in polyline.iter().rev().tuple_windows() {
        let circ_xy = Point2::from(m2.translation.vector);
        if let Some(contact_xy) = circle_line_contact(circ_xy, g2.radius, m1 * a, m1 * b) {
            return Some(contact_xy);
        }
    }
    None
}

// https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/PurePursuit.html
impl SideControlTrait for PurePursuitPolicy {
    fn choose_steer(&mut self, road: &Road, car_i: usize, trajectory: &[Point2<f64>]) -> f64 {
        // if car_i == 0 {
        //     return PI / 4.0;
        // }

        let car = &road.cars[car_i];
        // let car_rear_x = car.x - car.length * car.theta.cos();
        // let car_rear_y = car.y - car.length * car.theta.sin();

        // if !road.is_truth && trajectory.len() == 3 {
        //     let pt1 = trajectory[1];
        //     let pt2 = trajectory[2];

        //     if pt1.y == pt2.y && (car.y() - pt1.y).abs() < 0.1 {
        //         // short-circuit since we are going straight-enough
        //         return 0.0;
        //     }
        // }

        let car_ref_x = car.x();
        let car_ref_y = car.y();

        let target_ahead_dist = (self.ahead_time * car.vel)
            .min(AHEAD_DIST_MAX)
            .max(AHEAD_DIST_MIN);

        let contact = polyline_contact(
            &Isometry::identity(),
            trajectory,
            &Isometry::translation(car_ref_x, car_ref_y),
            &Ball::new(target_ahead_dist),
            target_ahead_dist * 2.0,
        );
        if contact.is_none() {
            eprintln_f!("{car_i=}, trajectory: {:.2?}", trajectory);
            eprintln_f!("{car_ref_x=:.2}, {car_ref_y=:.2}, {target_ahead_dist=:.2}");
        }

        let contact = contact.unwrap();

        let (target_x, target_y) = (contact[0], contact[1]);

        let car_to_target_x = target_x - car_ref_x;
        let car_to_target_y = target_y - car_ref_y;

        let ahead_dist = car_to_target_y.hypot(car_to_target_x);
        // let angle_to_target = car_to_target_y.atan2(car_to_target_x);
        let angle_to_target_sin = car_to_target_y / ahead_dist;

        // if car_i == 0 {
        //     eprintln_f!("{target_ahead_dist=:.2}, {target_x=:.2}, {target_y=:.2}");
        // }

        let target_steer = (2.0 * car.length * angle_to_target_sin / ahead_dist).atan();

        if self.debug_info.is_none() && road.debug && car.is_ego() {
            self.debug_info = Some(Box::new(PurePursuitPolicyDebug::new()));
        }

        if let Some(info) = self.debug_info.as_mut() {
            info.target_x = target_x;
            info.target_y = target_y;
            info.car_x = car_ref_x;
            info.car_y = car_ref_y;
            info.ahead_dist = target_ahead_dist;
            info.trajectory.clear();
            info.trajectory.extend_from_slice(trajectory);
        }

        target_steer
    }

    fn draw(&self, r: &mut Rvx) {
        if let Some(ref info) = self.debug_info {
            r.draw(
                Rvx::circle()
                    .scale(0.5)
                    .translate(&[info.target_x, info.target_y])
                    .color(RvxColor::WHITE),
            );

            r.draw(
                Rvx::circle()
                    .scale(info.ahead_dist)
                    .translate(&[info.car_x, info.car_y])
                    .color(RvxColor::WHITE.set_a(0.5)),
            );

            for (a, b) in info.trajectory.iter().tuple_windows() {
                r.draw(Rvx::line([a.x, a.y, b.x, b.y], 2.0).color(RvxColor::WHITE));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use parry2d_f64::{math::Point, na::Vector2};

    #[test]
    fn test_polyline_contact1() {
        let contact = polyline_contact(
            &Isometry::identity(),
            &vec![Point::new(0.0, 0.0), Point::new(20.0, 0.0)],
            &Isometry::identity(),
            &Ball::new(10.0),
            0.0,
        )
        .unwrap();

        eprintln!("contact p: {:.2}", contact);

        assert_eq!(contact, Point::new(10.0, 0.0));
    }

    #[test]
    fn test_polyline_contact2() {
        let contact = polyline_contact(
            &Isometry::identity(),
            &vec![Point::new(0.0, 0.0), Point::new(0.0, 20.0)],
            &Isometry::identity(),
            &Ball::new(10.0),
            0.0,
        )
        .unwrap();

        eprintln!("contact p: {:.2}", contact);

        assert_eq!(contact, Point::new(0.0, 10.0));
    }

    #[test]
    #[should_panic]
    fn test_parry_polyline_contact() {
        let contact = parry2d_f64::query::contact(
            &Isometry::identity(),
            &parry2d_f64::shape::Polyline::new(
                vec![
                    Point::new(0.0, 0.0),
                    Point::new(20.0, 0.0),
                    Point::new(20.0, 0.01),
                    Point::new(0.0, 0.01),
                ],
                None,
            ),
            &Isometry::identity(),
            &parry2d_f64::shape::Ball::new(10.0),
            1000.0,
        )
        .unwrap()
        .unwrap();

        eprintln!("{:?}", contact);

        assert_eq!(contact.point1, Point::new(10.0, 0.0));
    }

    #[test]
    fn test_contact_example() {
        let contact = parry2d_f64::query::contact(
            &Isometry::translation(1.0, 1.0),
            &parry2d_f64::shape::Ball::new(1.0),
            &Isometry::identity(),
            &parry2d_f64::shape::Cuboid::new(Vector2::new(1.0, 1.0)),
            1.0,
        )
        .unwrap()
        .unwrap();

        assert!(contact.dist < 0.0);
    }

    #[test]
    fn test_contact_squares() {
        let contact = parry2d_f64::query::contact(
            // &Ball::new(1.0),
            &Isometry::identity(),
            &parry2d_f64::shape::Cuboid::new(Vector2::new(0.5, 0.5)),
            &Isometry::translation(0.9, 0.0),
            &parry2d_f64::shape::Cuboid::new(Vector2::new(0.5, 0.5)),
            0.0,
        )
        .unwrap()
        .unwrap();

        eprintln!(
            "p1: {:.2}, p2: {:.2}, dist: {:.2}",
            contact.point1, contact.point2, contact.dist
        );

        // assert_eq!(contact.point1, Point::new(0.0, 0.0));
    }

    #[test]
    fn test_circ_line_intersection() {
        let txy = Vector2::new(10.0, 10.0);
        let circ_xy = Point2::new(0.0, 0.0) + txy;

        let p1 = Point2::new(-2.0, 0.0) + txy;
        let p2 = Point2::new(2.0, 0.0) + txy;
        assert_eq!(
            circle_line_contact(circ_xy, 1.0, p1, p2),
            Some(Point2::new(-1.0, 0.0) + txy)
        );

        let p1 = Point2::new(-1.0, 0.0) + txy;
        let p2 = Point2::new(1.0, 0.0) + txy;
        assert_eq!(
            circle_line_contact(circ_xy, 1.0, p1, p2),
            Some(Point2::new(-1.0, 0.0) + txy)
        );

        let p1 = Point2::new(-10.0, 0.0) + txy;
        let p2 = Point2::new(10.0, 0.0) + txy;
        assert_eq!(
            circle_line_contact(circ_xy, 1.0, p1, p2),
            Some(Point2::new(-1.0, 0.0) + txy)
        );

        let p1 = Point2::new(0.0, -2.0) + txy;
        let p2 = Point2::new(0.0, 2.0) + txy;
        assert_eq!(
            circle_line_contact(circ_xy, 1.0, p1, p2),
            Some(Point2::new(0.0, -1.0) + txy)
        );

        let p1 = Point2::new(0.0, 2.0) + txy;
        let p2 = Point2::new(0.0, -2.0) + txy;
        assert_eq!(
            circle_line_contact(circ_xy, 1.0, p1, p2),
            Some(Point2::new(0.0, 1.0) + txy)
        );

        // let p1 = Point2::new(0.0, -1.0) + txy;
        // let p2 = Point2::new(0.0, 1.0) + txy;
        // assert_eq!(
        //     circle_line_contact(circ_xy, 10.0, p1, p2),
        //     Some(Point2::new(0.0, -1.0) + txy)
        // );

        let p1 = Point2::new(-2.0, 2.0);
        let p2 = Point2::new(2.0, -2.0);
        assert_relative_eq!(
            circle_line_contact(Point2::new(0.0, 0.0), 1.0, p1, p2).unwrap(),
            Point2::new(-2.0f64.sqrt() / 2.0, 2.0f64.sqrt() / 2.0)
        );

        let p1 = Point2::new(-2.0, -2.0);
        let p2 = Point2::new(2.0, 2.0);
        assert_relative_eq!(
            circle_line_contact(Point2::new(0.0, 0.0), 1.0, p1, p2).unwrap(),
            Point2::new(-2.0f64.sqrt() / 2.0, -2.0f64.sqrt() / 2.0)
        );

        let p1 = Point2::new(0.5, -2.0);
        let p2 = Point2::new(0.5, 2.0);
        assert_relative_eq!(
            circle_line_contact(Point2::new(0.0, 0.0), 1.0, p1, p2).unwrap(),
            Point2::new(0.5, -(1.0 - 0.5f64.powi(2)).sqrt())
        );

        let p1 = Point2::new(0.5, 2.0);
        let p2 = Point2::new(0.5, -2.0);
        assert_relative_eq!(
            circle_line_contact(Point2::new(0.0, 0.0), 1.0, p1, p2).unwrap(),
            Point2::new(0.5, (1.0 - 0.5f64.powi(2)).sqrt())
        );
    }

    #[test]
    fn test_circ_line_no_intersection() {
        let txy = Vector2::new(10.0, 10.0);
        let circ_xy = Point2::new(0.0, 0.0) + txy;

        let p1 = Point2::new(-2.0, 1.1) + txy;
        let p2 = Point2::new(2.0, 1.1) + txy;
        assert_eq!(circle_line_contact(circ_xy, 1.0, p1, p2), None);

        let p1 = Point2::new(-1.0, -1.1) + txy;
        let p2 = Point2::new(1.0, -1.1) + txy;
        assert_eq!(circle_line_contact(circ_xy, 1.0, p1, p2), None);

        let p1 = Point2::new(-2.0, -2.0) + txy;
        let p2 = Point2::new(1.0, 100.0) + txy;
        assert_eq!(circle_line_contact(circ_xy, 1.0, p1, p2), None);

        let p1 = Point2::new(-1.1, -1.0) + txy;
        let p2 = Point2::new(-1.1, 1.0) + txy;
        assert_eq!(circle_line_contact(circ_xy, 1.0, p1, p2), None);
    }

    #[test]
    fn test_circ_line_failure1() {
        let p1 = Point2::new(-7.5302913661567965, 2.4791392629445359);
        let p2 = Point2::new(-6.5, 2.5);
        let circ_xy = Point2::new(-8.2679391016533117, 2.3513395997966295);
        assert!(circle_line_contact(circ_xy, 0.2, p1, p2).is_none());
    }
}
