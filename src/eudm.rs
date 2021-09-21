use rand::prelude::StdRng;

use crate::{
    arg_parameters::Parameters,
    cost::Cost,
    delayed_policy::DelayedPolicy,
    mpdm::make_policy_choices,
    road::Road,
    road_set::RoadSet,
    road_set_for_scenario,
    side_policies::{SidePolicy, SidePolicyTrait},
};

fn dcp_tree_search(
    params: &Parameters,
    policy_choices: &[SidePolicy],
    roads: RoadSet,
    debug: bool,
) -> (Option<SidePolicy>, Vec<rvx::Shape>) {
    let mut traces = Vec::new();

    let unchanged_policy = roads.ego_policy();
    let operating_policy = unchanged_policy.operating_policy();
    let eudm = &params.eudm;

    if debug {
        eprintln!(
            "{}: EUDM DCP-Tree search policies and costs, starting with policy {}",
            roads.timesteps(),
            unchanged_policy.policy_id(),
        );
    }

    let max_car_traces_depth = 3;

    let mut best_sub_policy = None;
    let mut best_switch_depth = 0;
    let mut best_cost = Cost::max_value();

    // Let's first consider the ongoing policy, which may be mid-way through a transition
    // unlike everything else we will consider, which won't transition policies for at least some period
    {
        let mut ongoing_roads = roads.clone();
        for depth_level in 0..eudm.search_depth {
            if depth_level < max_car_traces_depth {
                ongoing_roads.reset_car_traces();
            } else {
                ongoing_roads.disable_car_traces();
            }
            ongoing_roads.take_update_steps(eudm.layer_t, eudm.dt);
            traces.append(&mut ongoing_roads.make_traces(depth_level, false));
        }
        let cost = ongoing_roads.cost();
        if debug {
            let unchanged_policy_id = unchanged_policy.policy_id();
            eprintln_f!(
                "Unchanged: {unchanged_policy_id}: {cost:7.2?} = {:7.2}, {unchanged_policy:?}",
                cost.total()
            );
        }
        if cost < best_cost {
            best_cost = cost;
            best_sub_policy = None;
        }
    }

    // this copy of the roads will be advanced by layer_t each time through the loop
    // to avoid doing duplicate work.
    let mut init_policy_roads = roads.clone();
    init_policy_roads.set_ego_policy(&operating_policy);

    let start_depth = if eudm.allow_different_root_policy {
        0
    } else {
        1
    };

    for switch_depth in start_depth..=eudm.search_depth {
        if switch_depth < max_car_traces_depth {
            init_policy_roads.reset_car_traces();
        } else {
            init_policy_roads.disable_car_traces();
        }

        if switch_depth > 0 {
            init_policy_roads.take_update_steps(eudm.layer_t, eudm.dt);
            traces.append(&mut init_policy_roads.make_traces(switch_depth - 1, false));
        }

        if switch_depth == eudm.search_depth {
            if debug {
                eprintln_f!(
                    "switch time: {}, {operating_policy:?}: {:7.2?} = {:7.2}",
                    switch_depth as f64 * eudm.layer_t,
                    init_policy_roads.cost(),
                    init_policy_roads.cost().total()
                );
            }

            let cost = init_policy_roads.cost();
            if cost < best_cost {
                best_cost = cost;
                best_switch_depth = switch_depth;
                best_sub_policy = Some(&operating_policy);
            }
        } else {
            for (i, sub_policy) in policy_choices.iter().enumerate() {
                let mut roads = init_policy_roads.clone();
                if sub_policy.policy_id() == operating_policy.policy_id() {
                    continue;
                }
                roads.set_ego_policy_not_switched(sub_policy);

                for depth_level in switch_depth..eudm.search_depth {
                    if depth_level < max_car_traces_depth {
                        roads.reset_car_traces();
                    } else {
                        roads.disable_car_traces();
                    }
                    roads.take_update_steps(eudm.layer_t, eudm.dt);
                    traces.append(&mut roads.make_traces(depth_level, false));
                }

                if debug {
                    eprintln_f!(
                        "switch time: {}, to {i}: {sub_policy:?}: {:7.2?} = {:7.2}",
                        switch_depth as f64 * eudm.layer_t,
                        roads.cost(),
                        roads.cost().total()
                    );
                }

                let cost = roads.cost();
                if cost < best_cost {
                    best_cost = cost;
                    best_switch_depth = switch_depth;
                    best_sub_policy = Some(sub_policy);
                }
            }
        }
    }

    // will be Some if we should switch policies after one layer, and None to stay the same
    if let Some(best_sub_policy) = best_sub_policy {
        if debug {
            eprintln_f!(
                "Choose policy with best_cost {:.2}, {best_switch_depth=}, and {best_sub_policy:?}",
                best_cost.total()
            );
        }
        (
            Some(SidePolicy::DelayedPolicy(DelayedPolicy::new(
                operating_policy.clone(),
                best_sub_policy.clone(),
                eudm.layer_t * best_switch_depth as f64,
            ))),
            traces,
        )
    } else {
        if debug {
            eprintln_f!("Choose to keep unchanged policy with {best_cost=:.2}");
        }
        (None, traces)
    }
}

pub fn dcp_tree_choose_policy(
    params: &Parameters,
    true_road: &Road,
    rng: &mut StdRng,
) -> (Option<SidePolicy>, Vec<rvx::Shape>) {
    let roads = road_set_for_scenario(params, true_road, rng, params.eudm.samples_n);
    let debug = params.policy_report_debug
        && true_road.debug
        && true_road.timesteps + params.debug_steps_before >= params.max_steps as usize;
    let policy_choices = make_policy_choices(params);
    dcp_tree_search(params, &policy_choices, roads, debug)
}
