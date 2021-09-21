use std::collections::BinaryHeap;

use itertools::Itertools;
use ordered_float::NotNan;

use crate::{
    arg_parameters::Parameters, belief::Belief, car::SPEED_LOW,
    mpdm::make_obstacle_vehicle_policy_belief_states, road::Road, road_set::RoadSet,
};

fn key_vehicles(params: &Parameters, road: &Road) -> Vec<(usize, f64)> {
    let ego = &road.cars[0];
    let dx_thresh = params.cfb.key_vehicle_base_dist
        + ego.vel.max(SPEED_LOW) * params.cfb.key_vehicle_dist_time;

    let mut car_ids = Vec::new();
    for c in &road.cars[1..] {
        if c.crashed {
            continue;
        }

        let dx = (ego.x() - c.x()).abs();
        // if params.cfb_debug && road.super_debug() {
        //     eprintln_f!("ego to {c.car_i}: {dx=:.2}, {dx_thresh=:.2}");
        // }
        if dx <= dx_thresh {
            car_ids.push((c.car_i, dx));
        }
    }

    car_ids
}

fn most_probable_cartesian_product_scenarios(
    car_is: &[usize],
    belief: &Belief,
    n_policies: usize,
    n_scenarios: usize,
) -> Vec<(f64, Vec<(usize, usize)>)> {
    let mut top_n_scenarios: BinaryHeap<(std::cmp::Reverse<NotNan<f64>>, Vec<(usize, usize)>)> =
        BinaryHeap::new();
    let mut current_scenario = car_is.iter().map(|a| (*a, 0)).collect_vec();
    'outer: loop {
        let probability: f64 = current_scenario
            .iter()
            .map(|(car_i, policy_i)| belief.get(*car_i, *policy_i))
            .product();
        let probability = NotNan::new(probability).unwrap();

        if top_n_scenarios.len() < n_scenarios {
            top_n_scenarios.push((std::cmp::Reverse(probability), current_scenario.clone()));
        } else if top_n_scenarios
            .peek()
            .map(|(min_p, _)| probability > min_p.0)
            .unwrap()
        {
            top_n_scenarios.pop();
            top_n_scenarios.push((std::cmp::Reverse(probability), current_scenario.clone()));
        }

        // increment the `current_scenario` to the next situation
        for scenario in current_scenario.iter_mut() {
            scenario.1 += 1;
            if scenario.1 < n_policies {
                continue 'outer;
            }
            scenario.1 = 0;
        }
        break;
    }
    let mut top_n_scenarios = top_n_scenarios
        .into_iter()
        .map(|(p, scenario)| (*p.0, scenario))
        .rev()
        .collect_vec();

    top_n_scenarios.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    top_n_scenarios
}

pub fn conditional_focused_branching(
    params: &Parameters,
    road: &Road,
    n: usize,
) -> (RoadSet, Vec<usize>) {
    let belief = road.belief.as_ref().unwrap();
    let debug = params.cfb_debug && road.super_debug();

    let key_car_ids = key_vehicles(params, road);
    if debug {
        eprintln_f!("{key_car_ids=:?}");
    }
    let uncertain_car_ids = key_car_ids
        .into_iter()
        .filter(|&(car_i, _dx)| belief.is_uncertain(car_i, params.cfb.uncertainty_threshold))
        .collect_vec();
    if debug {
        eprintln_f!("{uncertain_car_ids=:?}");
    }

    // For each car, perform an open-loop simulation with only that car, using each real policy.
    // I guess the ego-vehicle gets to keep using its real policy?
    // (And I guess tree search and mcts would both need to be able to produce policies that contain their full set of changes)
    // And then I guess all the other vehicles are just made to be constant-velocity?
    // Alternatively... maybe _no_ vehicles use the intelligent driver model? but then what keeps the ego vehicle in
    // a light acceleration from just speeding off into stuff? Or the simulation making any sense at all?
    // I imagine that the uncertain and the ego vehicle must follow their real closed-loop policies for this to do any good.

    let policies = make_obstacle_vehicle_policy_belief_states(params);

    let open_loop_sims = uncertain_car_ids
        .into_iter()
        .map(|(car_i, dx)| {
            let road = road.open_loop_estimate(car_i);
            let costs = policies
                .iter()
                .map(|policy| {
                    let mut road = road.clone();
                    road.cars[car_i].side_policy = Some(policy.clone());
                    road.car_traces = None;
                    road.take_update_steps(params.cfb.horizon_t, params.cfb.dt);
                    // eprintln_f!("{car_i=} {road.cost:.2?} {policy:?}");
                    road.cost.total()
                })
                .collect_vec();

            let worst_cost = *costs
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let best_cost = *costs
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let riskiness = worst_cost - best_cost;

            (car_i, riskiness, dx, costs)
        })
        .collect_vec();

    if debug {
        eprintln!("Open loop sim results:");
        for open_loop_sim in open_loop_sims.iter() {
            eprintln_f!("{open_loop_sim:.2?}");
        }
    }

    let mut sorted_open_sims = open_loop_sims;
    // descending by riskiness then ascending by dx
    sorted_open_sims.sort_by(|(_, risk_a, dx_a, _), (_, risk_b, dx_b, _)| {
        risk_b
            .partial_cmp(risk_a)
            .unwrap()
            .then_with(|| dx_a.partial_cmp(dx_b).unwrap())
    });

    if debug {
        eprintln!("Potentially dangerous sims:");
        for sim in sorted_open_sims.iter() {
            eprintln_f!("{sim:.2?}");
        }
    }

    sorted_open_sims.truncate(params.cfb.max_n_for_cartesian_product);

    let selected_important_car_ids = sorted_open_sims.iter().map(|a| a.0).collect_vec();

    if debug {
        eprintln!("Choosing to consider all permutations of:");
        for sim in sorted_open_sims.iter() {
            eprintln_f!("{sim:.2?}");
        }
    }

    let mut sim_road = road.sim_estimate();
    // Each car (besides ego) defaults to the policy that is most likely for it
    for c in sim_road.cars[1..].iter_mut() {
        let policy_i = belief.get_most_likely(c.car_i);
        c.side_policy = Some(policies[policy_i].clone());
    }

    let top_n_scenarios = most_probable_cartesian_product_scenarios(
        &selected_important_car_ids,
        belief,
        policies.len(),
        n,
    );

    // sort descending and choose just the most probable
    // ranked_scenarios.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // ranked_scenarios.truncate(n);
    let mut roads = top_n_scenarios
        .into_iter()
        .map(|(prob, scenario)| {
            let mut sim_road = sim_road.clone();
            for (car_i, policy_i) in scenario.iter() {
                sim_road.cars[*car_i].side_policy = Some(policies[*policy_i].clone());
            }

            sim_road.cost.weight = prob;
            sim_road
        })
        .collect_vec();

    if roads.is_empty() {
        roads.push(sim_road);
    }

    (RoadSet::new(roads), selected_important_car_ids)
}

#[cfg(test)]
mod tests {
    use crate::belief::Belief;

    use super::*;

    #[test]
    fn most_probable_cartesian_product() {
        let n_cars = 5;
        let policies = vec![0, 1, 2];
        let risky_car_is = vec![2, 3, 4];

        let beliefs = vec![
            Belief::for_all_cars(n_cars, &[0.1, 0.2, 0.3]),
            Belief::for_all_cars(n_cars, &[0.3, 0.4, 0.1]),
            Belief::for_all_cars(n_cars, &[0.3, 0.2, 0.1]),
        ];

        for n_scenarios in 2..10 {
            for belief in beliefs.iter() {
                let scenarios = most_probable_cartesian_product_scenarios(
                    &risky_car_is,
                    belief,
                    policies.len(),
                    n_scenarios,
                );

                let scenarios_reference = most_probable_cartesian_product_reference(
                    &risky_car_is,
                    belief,
                    policies.len(),
                    n_scenarios,
                );

                let scenario_policy_sets = scenarios
                    .into_iter()
                    .map(|(p, scenario)| {
                        let mut scenario_policies = scenario
                            .into_iter()
                            .map(|(_, policy_i)| policy_i)
                            .collect_vec();
                        scenario_policies.sort();
                        (p, scenario_policies)
                    })
                    .collect_vec();

                let scenario_reference_policy_sets = scenarios_reference
                    .into_iter()
                    .map(|(p, scenario)| {
                        let mut scenario_policies = scenario
                            .into_iter()
                            .map(|(_, policy_i)| policy_i)
                            .collect_vec();
                        scenario_policies.sort();
                        (p, scenario_policies)
                    })
                    .collect_vec();

                assert_eq!(scenario_policy_sets, scenario_reference_policy_sets);
            }
        }
    }

    fn most_probable_cartesian_product_reference(
        car_is: &[usize],
        belief: &Belief,
        n_policies: usize,
        n_scenarios: usize,
    ) -> Vec<(f64, Vec<(usize, usize)>)> {
        let risky_car_policies = car_is
            .iter()
            .map(|car_i| (0..n_policies).map(move |p_i| (*car_i, p_i)));
        let mut ranked_scenarios = Vec::new();
        let scenarios = risky_car_policies.multi_cartesian_product().collect_vec();
        for scenario in scenarios.iter() {
            let probability: f64 = scenario
                .iter()
                .map(|(car_i, policy_i)| belief.get(*car_i, *policy_i))
                .product();

            ranked_scenarios.push((probability, scenario))
        }
        ranked_scenarios.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        ranked_scenarios.truncate(n_scenarios);
        let ranked_scenarios = ranked_scenarios
            .into_iter()
            .map(|(p, scenario)| (p, scenario.clone()))
            .collect_vec();
        ranked_scenarios
    }
}
