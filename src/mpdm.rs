use rand::prelude::StdRng;

use crate::{
    arg_parameters::Parameters,
    cost::Cost,
    lane_change_policy::{LaneChangePolicy, LongitudinalPolicy},
    road::Road,
    road_set::RoadSet,
    road_set_for_scenario,
    side_policies::{SidePolicy, SidePolicyTrait},
};

pub fn make_obstacle_vehicle_policy_choices(params: &Parameters) -> Vec<SidePolicy> {
    let mut policy_choices = Vec::new();

    for lane_i in [0, 1] {
        for long_policy in [LongitudinalPolicy::Maintain, LongitudinalPolicy::Accelerate] {
            policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                policy_choices.len() as u32,
                Some(lane_i),
                params.lane_change_time,
                true,
                long_policy,
            )));
        }
    }

    policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
        policy_choices.len() as u32,
        None,
        params.lane_change_time,
        true,
        LongitudinalPolicy::Decelerate,
    )));

    policy_choices
}

pub fn make_obstacle_vehicle_policy_belief_states(params: &Parameters) -> Vec<SidePolicy> {
    let mut policy_choices = Vec::new();

    for lane_i in [0, 1] {
        for long_policy in [LongitudinalPolicy::Maintain, LongitudinalPolicy::Accelerate] {
            for wait_for_clear in [false, true] {
                policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                    policy_choices.len() as u32,
                    Some(lane_i),
                    params.lane_change_time,
                    wait_for_clear,
                    long_policy,
                )));
            }
        }
    }

    policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
        policy_choices.len() as u32,
        None,
        params.lane_change_time,
        true,
        LongitudinalPolicy::Decelerate,
    )));

    policy_choices
}

pub fn make_policy_choices(params: &Parameters) -> Vec<SidePolicy> {
    let mut policy_choices = Vec::new();

    let long_policies = vec![LongitudinalPolicy::Maintain, LongitudinalPolicy::Accelerate];

    for &lane_i in &[0, 1] {
        for &long_policy in long_policies.iter() {
            policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
                policy_choices.len() as u32,
                Some(lane_i),
                params.lane_change_time,
                false,
                long_policy,
            )));
        }
    }

    policy_choices.push(SidePolicy::LaneChangePolicy(LaneChangePolicy::new(
        policy_choices.len() as u32,
        None,
        params.lane_change_time,
        false,
        LongitudinalPolicy::Decelerate,
    )));

    policy_choices
}

fn evaluate_policy(
    params: &Parameters,
    roads: &RoadSet,
    policy: &SidePolicy,
) -> (Cost, Vec<rvx::Shape>) {
    let mut roads = roads.clone();
    roads.set_ego_policy(policy);

    let mpdm = &params.mpdm;
    roads.reset_car_traces();
    roads.take_update_steps(mpdm.forward_t, mpdm.dt);

    (roads.cost(), roads.make_traces(0, false))
}

pub fn mpdm_choose_policy(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
) -> (Option<SidePolicy>, Vec<rvx::Shape>) {
    let mut traces = Vec::new();
    let roads = road_set_for_scenario(params, true_road, rng, params.mpdm.samples_n);
    let debug = params.policy_report_debug
        && true_road.debug
        && true_road.timesteps + params.debug_steps_before >= params.max_steps as usize;
    if debug {
        eprintln!(
            "{}: MPDM search policies and costs, starting with policy {}",
            roads.timesteps(),
            roads.ego_policy().policy_id(),
        );
        eprintln!(
            "Starting from base costs: {:7.2?} = {:7.2}",
            roads.cost(),
            roads.cost().total()
        );
    }

    let policy_choices = make_policy_choices(params);
    let mut best_cost = Cost::max_value();
    let mut best_policy = None;

    for (i, policy) in policy_choices.into_iter().enumerate() {
        // if roads.timesteps() >= 2200 && i != 3 {
        //     continue;
        // }
        // if i == 0 || i == 3 {
        //     continue;
        // }

        let (cost, mut new_traces) = evaluate_policy(params, &roads, &policy);
        traces.append(&mut new_traces);
        // eprint!("{:.2} ", cost);
        // eprintln!("{:?}: {:.2} ", policy, cost);
        if debug {
            eprintln_f!("{i}: {policy:?}: {:7.2?} = {:7.2}", cost, cost.total());
        }

        if cost < best_cost {
            best_cost = cost;
            best_policy = Some(policy);
        }
    }
    // eprintln!();

    (best_policy, traces)
}
