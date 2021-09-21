mod arg_parameters;
mod parameters_sql;
mod problem_scenario;

use arg_parameters::{run_parallel_scenarios, Parameters};
#[allow(unused)]
use fstrings::{eprintln_f, format_args_f, println_f, write_f};
use itertools::Itertools;
use problem_scenario::{ProblemScenario, Simulator};
use progressive_mcts::cost_set::CostSet;
use progressive_mcts::klucb::klucb_bernoulli;
use progressive_mcts::{ChildSelectionMode, CostBoundMode};
use rand::{
    prelude::{SliceRandom, StdRng},
    SeedableRng,
};

#[derive(Clone, Copy, Debug)]
pub struct RunResults {
    steps_taken: usize,
    chosen_cost: f64,
    chosen_true_cost: f64,
    true_best_cost: f64,
    regret: f64,
    cost_estimation_error: f64,
    sum_repeated: usize,
}

impl std::fmt::Display for RunResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "{s.steps_taken:6} {s.chosen_cost:7.2} {s.chosen_true_cost:7.2} {s.true_best_cost:7.2} {s.sum_repeated}"
        )
    }
}

fn compute_selection_index(
    params: &Parameters,
    total_n: f64,
    ln_total_n: f64,
    n_trials: usize,
    cost: f64,
    mode: ChildSelectionMode,
    variance: Option<f64>,
) -> Option<f64> {
    if n_trials == 0 {
        return None;
    }

    let mean_cost = cost;
    let n = n_trials as f64;
    let ln_t_over_n = ln_total_n / n;
    let index = match mode {
        ChildSelectionMode::UCB => {
            let upper_margin = params.ucb_const * ln_t_over_n.sqrt();
            assert!(upper_margin.is_finite(), "{}", n);
            mean_cost + upper_margin
        }
        ChildSelectionMode::UCBV => {
            let variance = variance.unwrap();
            let upper_margin = params.ucb_const
                * (params.ucbv_const * (variance * ln_t_over_n).sqrt() + ln_t_over_n);
            mean_cost + upper_margin
        }
        ChildSelectionMode::UCBd => {
            let a = (1.0 + n) / (n * n);
            let b = (total_n * (1.0 + n).sqrt() / params.ucbd_const).ln();
            let upper_margin = params.ucb_const * (a * (1.0 + 2.0 * b)).sqrt();
            if !upper_margin.is_finite() {
                eprintln_f!("{a=} {b=} {upper_margin=} {n=} {total_n=}");
                panic!();
            }
            mean_cost + upper_margin
        }
        ChildSelectionMode::KLUCB => {
            let scaled_mean = (1.0 - mean_cost / params.klucb_max_cost).min(1.0).max(0.0);
            let index = -klucb_bernoulli(scaled_mean, params.ucb_const.abs() * ln_t_over_n);
            index
        }
        ChildSelectionMode::KLUCBP => {
            let scaled_mean = (1.0 - mean_cost / params.klucb_max_cost).min(1.0).max(0.0);
            let index =
                -klucb_bernoulli(scaled_mean, params.ucb_const.abs() * (total_n / n).ln() / n);
            index
        }
        ChildSelectionMode::Uniform => n,
    };
    Some(index)
}

struct MctsNode<'a> {
    params: &'a Parameters,
    policy_choices: &'a [u32],

    policy: Option<u32>,
    depth: u32,
    n_trials: usize,
    expected_cost: Option<f64>,
    expected_cost_std_dev: Option<f64>,
    intermediate_costs: CostSet,
    marginal_costs: CostSet,

    seen_particles: Vec<bool>,
    n_particles_repeated: usize,

    sub_nodes: Option<Vec<MctsNode<'a>>>,
    costs: CostSet<f64, Option<Simulator<'a>>>,
    sub_node_repeated_particles: Vec<(f64, Simulator<'a>)>,
}

impl<'a> MctsNode<'a> {
    fn has_seen_particle(&self, i: usize) -> bool {
        if self.seen_particles.len() <= i {
            return false;
        }
        self.seen_particles[i]
    }

    fn seen_particle(&mut self, i: usize, seen: bool) {
        if self.seen_particles.len() <= i {
            self.seen_particles.resize(i + 1, false);
        }
        self.seen_particles[i] = seen;
    }

    // expand node?
    fn get_or_expand_sub_nodes_mut(&mut self) -> &mut Vec<MctsNode<'a>> {
        let params = self.params;
        if self.sub_nodes.is_none() {
            let policy_choices = self.policy_choices;

            self.sub_nodes = Some(
                policy_choices
                    .iter()
                    .map(|p| MctsNode {
                        params,
                        policy_choices,
                        policy: Some(p.clone()),
                        depth: self.depth + 1,
                        n_trials: 0,
                        expected_cost: None,
                        expected_cost_std_dev: None,
                        intermediate_costs: CostSet::new(),
                        marginal_costs: CostSet::new(),
                        seen_particles: vec![false; params.samples_n],
                        n_particles_repeated: 0,
                        sub_nodes: None,
                        costs: CostSet::new(),
                        sub_node_repeated_particles: Vec::new(),
                    })
                    .collect(),
            );
        }

        self.sub_nodes.as_mut().unwrap()
    }

    fn get_or_expand_sub_nodes(&mut self) -> &Vec<MctsNode<'a>> {
        self.get_or_expand_sub_nodes_mut()
    }

    fn variance(&self) -> f64 {
        self.costs.std_dev().powi(2)
    }

    fn min_child_expected_cost_and_std_dev(&self) -> Option<(f64, f64)> {
        self.sub_nodes.as_ref().and_then(|nodes| {
            nodes
                .iter()
                .filter_map(|n| Some((n.expected_cost?, n.expected_cost_std_dev?)))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
        })
    }

    fn mean_cost(&self) -> f64 {
        self.costs.mean()
    }

    fn std_dev_of_mean(&self) -> f64 {
        if self.costs.is_empty() {
            0.0
        } else {
            self.costs.std_dev() / (self.costs.len() as f64).sqrt()
        }
    }

    fn intermediate_cost(&self) -> f64 {
        if self.intermediate_costs.is_empty() {
            0.0
        } else {
            self.intermediate_costs.mean()
        }
    }

    fn intermediate_cost_std_dev(&self) -> f64 {
        if self.intermediate_costs.is_empty() {
            0.0
        } else {
            self.intermediate_costs.std_dev() / (self.intermediate_costs.len() as f64).sqrt()
        }
    }

    fn marginal_cost(&self) -> f64 {
        if self.marginal_costs.is_empty() {
            0.0
        } else {
            self.marginal_costs.mean()
        }
    }

    fn marginal_cost_std_dev(&self) -> f64 {
        if self.marginal_costs.is_empty() {
            0.0
        } else {
            self.marginal_costs.std_dev() / (self.marginal_costs.len() as f64).sqrt()
        }
    }

    fn compute_expected_cost_index(&self, total_n: f64, ln_total_n: f64) -> Option<f64> {
        let variance = if self.params.selection_mode == ChildSelectionMode::UCBV {
            Some(self.variance())
        } else {
            None
        };

        compute_selection_index(
            self.params,
            total_n,
            ln_total_n,
            self.costs.len(),
            self.expected_cost.unwrap(),
            self.params.selection_mode,
            variance,
        )
    }

    fn update_expected_cost(&mut self, bound_mode: CostBoundMode) {
        let (expected_cost, std_dev) = match bound_mode {
            CostBoundMode::Classic => (self.mean_cost(), self.std_dev_of_mean()),
            CostBoundMode::Expectimax => self
                .min_child_expected_cost_and_std_dev()
                .unwrap_or((self.mean_cost(), self.std_dev_of_mean())),
            CostBoundMode::LowerBound => {
                let (mut expected_cost, mut std_dev) = self
                    .min_child_expected_cost_and_std_dev()
                    .unwrap_or((0.0, 0.0));
                let intermediate_cost = self.intermediate_cost();
                if intermediate_cost > expected_cost {
                    expected_cost = intermediate_cost;
                    std_dev = self.intermediate_cost_std_dev();
                }
                (expected_cost, std_dev)
            }
            CostBoundMode::Marginal => {
                let (mut expected_cost, mut std_dev) = self
                    .min_child_expected_cost_and_std_dev()
                    .unwrap_or((0.0, 0.0));
                expected_cost += self.marginal_cost();
                std_dev = std_dev.hypot(self.marginal_cost_std_dev());
                (expected_cost, std_dev)
            }
            CostBoundMode::Same => panic!("Bound mode cannot be 'Same'"),
        };
        self.expected_cost = Some(expected_cost);
        self.expected_cost_std_dev = Some(std_dev);
    }

    fn get_best_policy_by_cost(&self) -> u32 {
        let chosen_policy = self
            .sub_nodes
            .as_ref()
            .unwrap()
            .iter()
            .min_by(|a, b| {
                let cost_a = a.expected_cost.unwrap_or(f64::MAX);
                let cost_b = b.expected_cost.unwrap_or(f64::MAX);
                cost_a.partial_cmp(&cost_b).unwrap()
            })
            .unwrap()
            .policy
            .unwrap();
        chosen_policy
    }

    fn get_best_policy_by_visits(&self) -> u32 {
        let chosen_policy = self
            .sub_nodes
            .as_ref()
            .unwrap()
            .iter()
            .max_by(|a, b| a.costs.len().cmp(&b.costs.len()))
            .unwrap()
            .policy
            .unwrap();
        chosen_policy
    }
}

fn find_trial_path(node: &mut MctsNode, rng: &mut StdRng, mut path: Vec<usize>) -> Vec<usize> {
    let params = node.params;

    let sub_depth = node.depth + 1;
    if sub_depth > params.search_depth {
        return path;
    } else {
        let n_trials = node.n_trials;
        let sub_nodes = node.get_or_expand_sub_nodes_mut();

        // choose a node to recurse down into!

        // choose any unexplored branch
        let unexplored = sub_nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.n_trials == 0)
            .map(|(i, _)| (sub_nodes[i].marginal_cost(), i))
            .collect_vec();
        if unexplored.len() > 0 {
            let sub_node_i = unexplored.choose(rng).unwrap().1;
            path.push(sub_node_i);
            return find_trial_path(&mut sub_nodes[sub_node_i], rng, path);
        }

        // Everything has been explored at least once: UCB time!
        let total_n = n_trials as f64;
        let ln_t = total_n.ln();
        let (_best_ucb, chosen_i) = sub_nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let index = node.compute_expected_cost_index(total_n, ln_t).unwrap();
                (index, i)
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        path.push(chosen_i);
        return find_trial_path(&mut sub_nodes[chosen_i], rng, path);
    }
}

fn should_replay_particle_at<'a>(
    node: &MctsNode<'a>,
    sub_node_i: usize,
) -> Option<(u32, f64, Simulator<'a>)> {
    if node.depth > 0 {
        return None;
    }

    let sub_node = &node.sub_nodes.as_ref().unwrap()[sub_node_i];

    // Prioritize repeating particles that have already been repeated by other sub nodes
    if let Some((c, sim)) = node
        .sub_node_repeated_particles
        .iter()
        .filter(|(_c, sim)| !sub_node.has_seen_particle(sim.particle.id))
        .nth(0)
    {
        assert_eq!(sim.depth, node.depth);
        assert!(node.depth < 4);
        return Some((sub_node.depth, *c, sim.clone()));
    }

    if let Some((c, sim)) = node
        .costs
        .iter()
        .filter(|(_c, sim)| {
            let sim = sim.as_ref().unwrap();
            !sub_node.has_seen_particle(sim.particle.id)
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
    {
        let sim = sim.as_ref().unwrap();
        assert_eq!(sim.depth, node.depth);
        assert!(node.depth < 4);
        return Some((sub_node.depth, *c, sim.clone()));
    }

    None
}

fn should_replay_particle<'a>(
    node: &MctsNode<'a>,
    path: &[usize],
) -> Option<(u32, f64, Simulator<'a>)> {
    if node.params.repeat_const <= 0.0 {
        return None;
    }
    let repeat_n = (node.params.repeat_const / (node.params.samples_n as f64)) as usize;
    if node.n_particles_repeated >= repeat_n {
        return None;
    }

    let mut node = node;
    let mut path = path;

    // we don't go to the very end of the path,
    // because at that point, there is no particle replaying to do!
    while path.len() >= 2 {
        let sub_node_i = path[0];
        let should_replay = should_replay_particle_at(node, sub_node_i);
        if should_replay.is_some() {
            return should_replay;
        }
        node = &node.sub_nodes.as_ref().unwrap()[sub_node_i];
        path = &path[1..];
    }
    None
}

fn find_and_run_trial<'a>(
    node: &mut MctsNode<'a>,
    sim: &mut Simulator<'a>,
    rng: &mut StdRng,
    steps_taken: &mut usize,
    n_completed: usize,
) -> f64 {
    let path = find_trial_path(node, rng, Vec::new());
    if let Some((depth, c, s)) = should_replay_particle(node, &path) {
        *sim = s.clone();

        assert_eq!(sim.depth + 1, depth);

        let score = run_trial(node, sim, rng, steps_taken, &path, depth as i32);

        for_node_in_path(node, &path[0..depth as usize - 1], |_| ())
            .sub_node_repeated_particles
            .push((c, s));

        let mut depth1_action = None;
        let final_node = for_node_in_path(node, &path[0..depth as usize + 1], |n| {
            if n.depth == 1 {
                depth1_action = Some(n.policy.unwrap());
            }
            n.n_particles_repeated += 1;
        });

        if final_node.params.is_single_run {
            eprintln_f!(
                "{n_completed}: {} Replaying particle {sim.particle.id:3} at depth {depth}",
                depth1_action.unwrap()
            );
        }

        return score;
    }

    let score = run_trial(node, sim, rng, steps_taken, &path, 0);

    if node.params.is_single_run {
        let mut depth1_action = None;
        for_node_in_path(node, &path[0..2], |n| {
            if n.depth == 1 {
                depth1_action = Some(n.policy.unwrap());
            }
        });
        eprintln_f!(
            "{n_completed}: {} Playing new particle {sim.particle.id:3}",
            depth1_action.unwrap()
        );
    }

    score
}

// calls f for each node in path, then returns the last node
fn for_node_in_path<'a, 'b, F>(
    node: &'a mut MctsNode<'b>,
    path: &[usize],
    mut f: F,
) -> &'a mut MctsNode<'b>
where
    F: FnMut(&mut MctsNode),
{
    let mut node = node;
    let mut path = path;
    while !path.is_empty() {
        f(node);
        node = &mut node.sub_nodes.as_mut().unwrap()[path[0]];
        path = &path[1..];
    }
    node
}

fn run_step<'a>(
    node: &mut MctsNode<'a>,
    sim: &mut Simulator<'a>,
    rng: &mut StdRng,
    steps_taken: &mut usize,
) -> Option<f64> {
    if let Some(ref policy) = node.policy {
        let prev_cost = sim.cost;
        sim.take_step(*policy, rng);
        node.intermediate_costs.push((sim.cost, ()));
        node.marginal_costs.push((sim.cost - prev_cost, ()));

        *steps_taken += 1;

        return Some(sim.cost);
    }
    None
}

fn run_trial<'a>(
    node: &mut MctsNode<'a>,
    sim: &mut Simulator<'a>,
    rng: &mut StdRng,
    steps_taken: &mut usize,
    path: &[usize],
    skip_depth: i32,
) -> f64 {
    let params = node.params;

    // skip over when we are repeating a particle and it has already been evaluated at this level
    let skip_over = skip_depth > 0;
    if !skip_over {
        run_step(node, sim, rng, steps_taken);
    }

    let orig_sim = sim.clone();

    let trial_final_cost = if path.is_empty() {
        assert_eq!(sim.depth, node.params.search_depth);
        sim.cost
    } else {
        run_trial(
            &mut node.sub_nodes.as_mut().unwrap()[path[0]],
            sim,
            rng,
            steps_taken,
            &path[1..],
            skip_depth - 1,
        )
    };

    if !skip_over {
        assert_eq!(node.depth, orig_sim.depth);
        node.costs.push((trial_final_cost, Some(orig_sim)));
        node.seen_particle(sim.particle.id, true);
        node.n_trials = node.costs.len();
    }

    node.update_expected_cost(params.bound_mode);

    trial_final_cost
}

fn print_report(
    scenario: &ProblemScenario,
    node: &MctsNode,
    parent_n_trials: f64,
    mut true_intermediate_cost: f64,
) {
    if node.n_trials > 0 {
        for _ in 0..node.depth {
            eprint!("    ");
        }
        let policy = node.policy.as_ref();
        let cost = node.expected_cost.unwrap();
        let std_dev = node.expected_cost_std_dev.unwrap();
        let mut additional_true_cost = 0.0;
        if let Some(dist_mean) = scenario.distribution.as_ref().map(|d| d.mean()) {
            additional_true_cost = dist_mean;
            true_intermediate_cost += additional_true_cost;
        }

        let _intermediate_cost = node.intermediate_cost();
        let marginal_cost = node.marginal_cost();
        let _variance = node.variance();

        let _costs_only = node.costs.iter().map(|(c, _)| *c).collect_vec();

        let index = node.compute_expected_cost_index(parent_n_trials, parent_n_trials.ln()).unwrap_or(99999.0);

        //  interm = {_intermediate_cost:6.1?}, \
        //  {node.intermediate_costs=:.2?}, \
        eprintln_f!(
            "n_trials: {node.n_trials}, {policy=:?}, {cost=:6.1}, {std_dev=:6.1}, \
             {index=:.3}, \
             marginal = {marginal_cost:6.1?}, \
             true = {additional_true_cost:6.1} ({true_intermediate_cost:6.1}), \
             marginal_costs = {:.2?}, \
             ",
            &node.marginal_costs.iter().map(|a| a.0).collect_vec()
            //  {_costs_only=:.2?}, \
            //  {node.costs=:.2?}" //,
        );
    }
    if let Some(sub_nodes) = &node.sub_nodes {
        for (policy_i, sub_node) in sub_nodes.iter().enumerate() {
            print_report(
                &scenario.children[policy_i],
                sub_node,
                node.n_trials as f64,
                true_intermediate_cost,
            );
        }
    }
}

fn true_best_child_cost(scenario: &ProblemScenario) -> (f64, f64, usize) {
    let add_cost = scenario.expected_marginal_cost();

    let best_child_cost = scenario
        .children
        .iter()
        .map(|c| {
            let a = true_best_child_cost(c);
            a.0 + a.1
        })
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, 0.0));

    (add_cost, best_child_cost.1, best_child_cost.0)
}

fn true_best_cost(scenario: &ProblemScenario, debug: bool) -> (f64, usize) {
    let (add_cost, best_child_cost, best_child_i) = true_best_child_cost(scenario);
    let total_cost = best_child_cost + add_cost;

    if debug {
        for _ in 0..scenario.depth {
            eprint!("    ");
        }
        eprintln_f!("{add_cost=:6.1} + {best_child_cost=:6.1} = {total_cost=:6.1}");

        for child in scenario.children.iter() {
            true_best_cost(child, debug);
        }
    }

    (total_cost, best_child_i)
}

fn set_final_choice_expected_values(params: &Parameters, node: &mut MctsNode) {
    if let Some(sub_nodes) = &mut node.sub_nodes {
        for sub_node in sub_nodes.iter_mut() {
            set_final_choice_expected_values(params, sub_node);
        }
    }

    if node.n_trials == 0 {
        return;
    }

    let final_choice_mode = if params.final_choice_mode == CostBoundMode::Same {
        params.bound_mode
    } else {
        params.final_choice_mode
    };

    node.update_expected_cost(final_choice_mode);
}

fn run_with_parameters(params: Parameters) -> RunResults {
    let policies = (0..params.n_actions).collect_vec();

    let mut node = MctsNode {
        params: &params,
        policy_choices: &policies,
        policy: None,

        depth: 0,
        n_trials: 0,
        expected_cost: None,
        expected_cost_std_dev: None,
        intermediate_costs: CostSet::new(),
        marginal_costs: CostSet::new(),
        seen_particles: vec![false; params.samples_n],
        n_particles_repeated: 0,

        sub_nodes: None,
        costs: CostSet::new(),
        sub_node_repeated_particles: Vec::new(),
    };

    let mut full_seed = [0; 32];
    full_seed[0..8].copy_from_slice(&params.rng_seed.to_le_bytes());
    let mut rng = StdRng::from_seed(full_seed);

    let scenario = ProblemScenario::new(params.search_depth, params.n_actions, &mut rng);

    let mut steps_taken = 0;

    // Expand first level so marginal_cost_confidence_interval has enough to go on
    node.get_or_expand_sub_nodes();

    let mut i = 0;
    loop {
        find_and_run_trial(
            &mut node,
            &mut Simulator::sample(&scenario, i, &mut rng),
            &mut rng,
            &mut steps_taken,
            i,
        );
        i += 1;

        if i >= params.samples_n {
            if params.most_visited_best_cost_consistency && i <= params.samples_n * 12 / 10 {
                // if we have this best policy inconsistency, do more trials to try to resolve it!
                let best_visits = node.get_best_policy_by_visits();
                let best_cost = node.get_best_policy_by_cost();
                if best_visits != best_cost {
                    if params.is_single_run {
                        eprintln_f!("{best_visits} != {best_cost}");
                    }
                    continue;
                }
            }
            break;
        }
    }

    if params.print_report {
        print_report(&scenario, &node, node.n_trials as f64, 0.0);
    }

    set_final_choice_expected_values(&params, &mut node);
    let chosen_policy = node.get_best_policy_by_cost();

    let chosen_true_cost = true_best_cost(&scenario.children[chosen_policy as usize], false).0;
    let (true_best_cost, _true_best_policy) = true_best_cost(&scenario, false);

    let mut sum_repeated = 0;
    if params.is_single_run {
        println_f!(
        "{chosen_policy=}: {node.expected_cost=:.2?}, {chosen_true_cost=:.2}, {true_best_cost=:.2}: {_true_best_policy=}");
    }

    for (i, sub_node) in node.sub_nodes.as_ref().unwrap().iter().enumerate() {
        if params.is_single_run {
            println_f!("{i}: {sub_node.n_particles_repeated=}");
        }
        sum_repeated += sub_node.n_particles_repeated;
    }
    if params.is_single_run {
        println_f!("steps taken: {steps_taken}");
        println_f!("total repeated: {sum_repeated}");
    }

    let chosen_cost = node.expected_cost.unwrap_or(99999.0);

    RunResults {
        steps_taken,
        chosen_cost,
        chosen_true_cost,
        true_best_cost,
        regret: chosen_true_cost - true_best_cost,
        cost_estimation_error: (chosen_cost - chosen_true_cost).abs(),
        sum_repeated,
    }
}

fn main() {
    run_parallel_scenarios();
}
