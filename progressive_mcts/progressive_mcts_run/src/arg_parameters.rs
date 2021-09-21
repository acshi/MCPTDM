use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{sync_channel, RecvTimeoutError},
        Arc, Mutex,
    },
    time::Duration,
};

use crate::parameters_sql::{
    create_table_sql, insert_sql, make_insert_specifiers, parse_parameters, specifier_params,
    specifiers_hash,
};
#[allow(unused)]
use fstrings::{format_args_f, format_f, println_f};
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{run_with_parameters, ChildSelectionMode, CostBoundMode};

#[derive(Clone, Debug)]
pub struct Parameters {
    pub search_depth: u32,
    pub n_actions: u32,
    pub ucb_const: f64,
    pub ucbv_const: f64,
    pub ucbd_const: f64,
    pub klucb_max_cost: f64,
    pub rng_seed: u64,
    pub samples_n: usize,

    pub bound_mode: CostBoundMode,
    pub final_choice_mode: CostBoundMode,
    pub selection_mode: ChildSelectionMode,
    pub repeat_const: f64,
    pub most_visited_best_cost_consistency: bool,

    pub thread_limit: usize,
    pub specifiers_hash: Option<i64>,

    pub print_report: bool,
    pub stats_analysis: bool,
    pub is_single_run: bool,
}

impl Parameters {
    fn new() -> Self {
        Self {
            search_depth: 4,
            n_actions: 5,
            ucb_const: -0.1, // for klucb, -1500 for UCB
            ucbv_const: 0.001,
            ucbd_const: 0.1,
            klucb_max_cost: 4700.0,
            rng_seed: 0,
            samples_n: 64,
            bound_mode: CostBoundMode::Marginal,
            final_choice_mode: CostBoundMode::Same,
            selection_mode: ChildSelectionMode::KLUCB,
            repeat_const: -1.0,
            most_visited_best_cost_consistency: true,

            thread_limit: 0,
            specifiers_hash: None,

            print_report: false,
            stats_analysis: false,
            is_single_run: false,
        }
    }
}

fn create_scenarios(
    base_p: &Parameters,
    name_value_pairs: &[(String, Vec<String>)],
) -> Vec<Parameters> {
    if name_value_pairs.is_empty() {
        return vec![base_p.clone()];
    }

    let mut scenarios = Vec::new();
    let (name, values) = &name_value_pairs[0];

    if name.starts_with("classic.") && base_p.bound_mode != CostBoundMode::Classic
        || name.starts_with("expectimax.") && base_p.bound_mode != CostBoundMode::Expectimax
        || name.starts_with("lower_bound.") && base_p.bound_mode != CostBoundMode::LowerBound
        || name.starts_with("marginal.") && base_p.bound_mode != CostBoundMode::Marginal
    {
        return create_scenarios(&base_p, &name_value_pairs[1..]);
    }

    if name.starts_with("ucb.") && base_p.selection_mode != ChildSelectionMode::UCB
        || name.starts_with("ucbv.") && base_p.selection_mode != ChildSelectionMode::UCBV
        || name.starts_with("ucbd.") && base_p.selection_mode != ChildSelectionMode::UCBd
        || name.starts_with("klucb.") && base_p.selection_mode != ChildSelectionMode::KLUCB
        || name.starts_with("klucb+.") && base_p.selection_mode != ChildSelectionMode::KLUCBP
    {
        return create_scenarios(&base_p, &name_value_pairs[1..]);
    }

    for value in values.iter() {
        let mut value_set = vec![value.to_owned()];

        // Do we have a numeric range? special-case handle that!
        let range_parts = value.split("-").collect_vec();
        if range_parts.len() == 2 {
            let low: Option<usize> = range_parts[0].parse().ok();
            let high: Option<usize> = range_parts[1].parse().ok();
            if let (Some(low), Some(high)) = (low, high) {
                if low < high {
                    value_set.clear();
                    for v in low..=high {
                        value_set.push(v.to_string());
                    }
                }
            }
        }

        for val in value_set {
            let mut params = base_p.clone();
            parse_parameters(&mut params, name, &val);
            if name_value_pairs.len() > 1 {
                scenarios.append(&mut create_scenarios(&params, &name_value_pairs[1..]));
            } else {
                scenarios.push(params);
            }
        }
    }

    scenarios
}

pub fn run_parallel_scenarios() {
    let parameters_default = Parameters::new();

    // let args = std::env::args().collect_vec();
    let mut name_value_pairs = Vec::<(String, Vec<String>)>::new();
    // let mut arg_i = 0;
    let mut name: Option<String> = None;
    let mut vals: Option<Vec<String>> = None;
    for arg in std::env::args()
        .skip(1)
        .chain(std::iter::once("::".to_owned()))
    {
        if arg == "--help" || arg == "help" {
            eprintln!("Usage: (<param name> [param value]* ::)*");
            eprintln!("For example: limit 8 12 16 24 32 :: steps 1000 :: rng_seed 0 1 2 3 4");
            eprintln!("Valid parameters and their default values:");
            let params_str = format!("{:?}", parameters_default)
                .replace(", file_name: None", "")
                .replace(", ", "\n\t")
                .replace("Parameters { ", "\t")
                .replace(" }", "");
            eprintln!("{}", params_str);
            std::process::exit(0);
        }
        if name.is_some() {
            if arg == "::" {
                let name = name.take().unwrap();
                if name_value_pairs.iter().any(|pair| pair.0 == name) {
                    panic!("Parameter {} has already been specified!", name);
                }
                name_value_pairs.push((name, vals.take().unwrap()));
            } else {
                vals.as_mut().unwrap().push(arg);
            }
        } else if arg != "::" {
            name = Some(arg);
            vals = Some(Vec::new());
        }
    }

    // for (name, vals) in name_value_pairs.iter() {
    //     eprintln!("{}: {:?}", name, vals);
    // }

    let base_scenario = parameters_default;
    let scenarios = create_scenarios(&base_scenario, &name_value_pairs);
    // for (i, scenario) in scenarios.iter().enumerate() {
    //     eprintln!("{}: {:?}", i, scenario.file_name);
    // }

    let n_scenarios = scenarios.len();
    eprintln!("Starting to run {} scenarios", n_scenarios);
    if n_scenarios == 0 {
        return;
    }

    let thread_limit = scenarios[0].thread_limit;
    if thread_limit > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_limit as usize)
            .build_global()
            .unwrap();
    }

    let n_scenarios_completed = AtomicUsize::new(0);

    let cache_filename = "results.db";
    let mut conn = rusqlite::Connection::open(cache_filename).unwrap();
    // create if doesn't exist (lazy way, ignoring an error)
    let _ = conn.execute(&create_table_sql(), []);

    let mut specifiers_hash_statement = conn
        .prepare("SELECT specifiers_hash FROM results;")
        .expect("prepare select specifiers_hash");
    let specifiers_hashs = specifiers_hash_statement
        .query_map([], |r| r.get::<_, i64>(0))
        .unwrap();
    let completed_result_set: BTreeSet<i64> = specifiers_hashs.filter_map(|a| a.ok()).collect();
    let completed_result_set = Mutex::new(completed_result_set);
    drop(specifiers_hash_statement);

    let many_scenarios = n_scenarios > 30000;
    if n_scenarios == 1 {
        let mut single_scenario = scenarios[0].clone();
        single_scenario.is_single_run = true;
        let res = run_with_parameters(single_scenario);
        println_f!("{res}");
    } else {
        let (tx, rx) = sync_channel(2048);
        let is_done = Arc::new(AtomicBool::new(false));

        let is_done_job = is_done.clone();
        let recv_thread = std::thread::spawn(move || loop {
            let transaction = conn.transaction().expect("transaction");
            let mut insert_statement = transaction.prepare(&insert_sql()).expect("prepare insert");

            let mut received_any = false;

            loop {
                match rx.recv_timeout(Duration::from_millis(1)) {
                    Ok((scenario, res)) => {
                        received_any = true;
                        let insert_specifiers = make_insert_specifiers(&scenario, &res);
                        insert_statement
                            .insert(specifier_params(&insert_specifiers).as_slice())
                            .expect("insert");
                    }
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }

            drop(insert_statement);
            transaction.commit().expect("commit");

            if !received_any && is_done_job.load(Ordering::Relaxed) {
                break;
            }
        });

        scenarios.par_iter().for_each(|scenario| {
            let result = std::panic::catch_unwind(|| {
            {
                let mut scenario = scenario.clone();
                scenario.specifiers_hash = Some(specifiers_hash(&scenario));

                if completed_result_set.lock().unwrap().contains(&scenario.specifiers_hash.unwrap()) {
                    n_scenarios_completed.fetch_add(1, Ordering::Relaxed);
                    return;
                }

                let res = run_with_parameters(scenario.clone());

                n_scenarios_completed.fetch_add(1, Ordering::Relaxed);
                if many_scenarios {
                    let completed = n_scenarios_completed.load(Ordering::Relaxed);
                    if completed % 500 == 0 {
                        println!(
                            "{}/{}: ",
                            n_scenarios_completed.load(Ordering::Relaxed),
                            n_scenarios
                        );
                    }
                } else {
                    print!(
                        "{}/{}: ",
                        n_scenarios_completed.load(Ordering::Relaxed),
                        n_scenarios
                    );
                    if scenario.stats_analysis {
                        println_f!(
                            "{res} {scenario.search_depth} {scenario.n_actions} {scenario.samples_n}"
                        );
                    } else {
                        println_f!("{res}");
                    }
                }

                // writeln_f!(file.lock().unwrap(), "{scenario_name} {res}").unwrap();
                // {
                //     let insert_specifiers = make_insert_specifiers(scenario, &res);
                //     let conn_guard = conn.lock().unwrap();
                //     let mut insert_statement =
                //         conn_guard.prepare(&insert_sql()).expect("prepare insert");
                //     insert_statement
                //         .insert(specifier_params(&insert_specifiers).as_slice())
                //         .expect("insert");
                // }
                tx.send((scenario, res)).expect("tx send");
            }
            });
            if result.is_err() {
                // eprintln!(
                //     "PANIC for scenario: {:?}",
                //     scenario.sc.as_ref().unwrap()
                // );
                // panic!();
                std::process::abort();
            }
        });

        is_done.store(true, Ordering::Relaxed);
        recv_thread.join().unwrap();
    }
}
