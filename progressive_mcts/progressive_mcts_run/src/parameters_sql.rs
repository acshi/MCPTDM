use std::{collections::hash_map::DefaultHasher, hash::Hasher};

use crate::{arg_parameters::Parameters, RunResults};
use itertools::Itertools;
use paste::paste;
use rusqlite::ToSql;

macro_rules! define_params {
    ($defining_type:ident, $($param:ident),*) => {
        paste! {
            // $(const [<$param:upper>]: &'static str = stringify!($param);)*
            // const $defining_type: &[&'static str] = &[$([<$param:upper>]),*];
            const [<$defining_type _PARAMS>]: &[&'static str] = &[$(stringify!($param)),*];

            fn [<parse_ $defining_type:lower _params>](params: &mut Parameters, name: &str, val: &str) -> bool {
                match name {
                    $(stringify!($param) => params.$param = val.parse().unwrap(),)*
                    _ => return false
                }
                true
            }

            fn [<hash_ $defining_type:lower _specifiers>](params: &Parameters, hasher: &mut DefaultHasher) {
                define_params!(@hasher hasher, $defining_type, params, $($param),*);
            }

            fn [<add_ $defining_type:lower _specifiers>](params: &Parameters, spec: &mut Vec<(&'static str, String)>) {
                $(spec.push((concat!(":", stringify!($param)), params.$param.to_string()));)*
            }

            const [<$defining_type _CREATE_TABLE_SQL>]: &'static str = stringify!($($param $defining_type),*);
            // const [<$defining_type _SELECT_WHERE_SQL>]: &'static str = concat!($(stringify!($param), " = :", stringify!($param), ", "),*);
        }
    };
    (@hasher $hasher:expr, INTEGER, $params:ident, $($param:ident),*) => {
        // $($hasher.write_u64($param as u64);)*
        use std::hash::Hash;
        $($params.$param.hash($hasher);)*
    };
    (@hasher $hasher:expr, TEXT, $params:ident, $($param:ident),*) => {
        // $($hasher.write_u64($param as u64);)*
        use std::hash::Hash;
        $($params.$param.hash($hasher);)*
    };
    (@hasher $hasher:expr, REAL, $params:ident, $($param:ident),*) => {
        $($hasher.write_u64($params.$param.to_bits());)*
    };
}

pub fn parse_parameters(params: &mut Parameters, name: &str, val: &str) {
    let name = name.split('.').last().unwrap();
    if parse_integer_params(params, name, val)
        || parse_text_params(params, name, val)
        || parse_real_params(params, name, val)
    {
        return;
    }
    match name {
        "thread_limit" => params.thread_limit = val.parse().unwrap(),
        "print_report" => params.print_report = val.parse().unwrap(),
        "stats_analysis" => params.stats_analysis = val.parse().unwrap(),
        _ => panic!("{} is not a valid parameter!", name),
    }
}

define_params!(
    INTEGER,
    rng_seed,
    search_depth,
    n_actions,
    samples_n,
    most_visited_best_cost_consistency
);

define_params!(TEXT, bound_mode, final_choice_mode, selection_mode);

define_params!(
    REAL,
    ucb_const,
    ucbv_const,
    ucbd_const,
    klucb_max_cost,
    repeat_const
);

macro_rules! define_result_values {
    ($($param:ident),*) => {
        const RESULT_VALUES: &[&'static str] = &[$(stringify!($param)),*];

        fn add_result_values(res: &RunResults, spec: &mut Vec<(&'static str, String)>) {
            $(spec.push((concat!(":", stringify!($param)), res.$param.to_string()));)*
        }

        const RESULT_CREATE_TABLE_SQL: &'static str = stringify!($($param REAL),*);
    };
}

define_result_values!(
    steps_taken,
    chosen_cost,
    chosen_true_cost,
    true_best_cost,
    regret,
    cost_estimation_error,
    sum_repeated
);

pub fn create_table_sql() -> String {
    format!(
        "CREATE TABLE results (id INTEGER PRIMARY KEY, specifiers_hash INTEGER, {}, {}, {}, {})",
        INTEGER_CREATE_TABLE_SQL,
        TEXT_CREATE_TABLE_SQL,
        REAL_CREATE_TABLE_SQL,
        RESULT_CREATE_TABLE_SQL
    )
}

// pub fn select_where_sql() -> String {
//     let mut sql = "SELECT id FROM results WHERE ".to_owned();
//     let mut added_any = false;
//     for param in INTEGER_PARAMS.iter().chain(TEXT_PARAMS).chain(REAL_PARAMS) {
//         if added_any {
//             sql.push_str(" AND ");
//         }
//         sql.push_str(param);
//         sql.push_str(" = :");
//         sql.push_str(param);
//         added_any = true;
//     }
//     sql
// }

pub fn insert_sql() -> String {
    let mut sql = "INSERT INTO results (specifiers_hash, ".to_owned();
    let mut added_any = false;
    for param in INTEGER_PARAMS
        .iter()
        .chain(TEXT_PARAMS)
        .chain(REAL_PARAMS)
        .chain(RESULT_VALUES)
    {
        if added_any {
            sql.push_str(", ");
        }
        sql.push_str(param);
        added_any = true;
    }

    sql.push_str(") VALUES (:specifiers_hash, ");

    let mut added_any = false;
    for param in INTEGER_PARAMS
        .iter()
        .chain(TEXT_PARAMS)
        .chain(REAL_PARAMS)
        .chain(RESULT_VALUES)
    {
        if added_any {
            sql.push_str(", ");
        }
        sql.push(':');
        sql.push_str(param);
        added_any = true;
    }

    sql.push_str(")");
    sql
}

pub fn make_select_specifiers(params: &Parameters) -> Vec<(&'static str, String)> {
    let mut spec = Vec::new();
    add_integer_specifiers(params, &mut spec);
    add_text_specifiers(params, &mut spec);
    add_real_specifiers(params, &mut spec);
    spec
}

pub fn specifiers_hash(params: &Parameters) -> i64 {
    let mut hasher = DefaultHasher::new();
    hash_integer_specifiers(params, &mut hasher);
    hash_text_specifiers(params, &mut hasher);
    hash_real_specifiers(params, &mut hasher);
    hasher.finish() as i64
}

pub fn make_insert_specifiers(
    params: &Parameters,
    results: &RunResults,
) -> Vec<(&'static str, String)> {
    let mut spec = make_select_specifiers(params);
    add_result_values(results, &mut spec);
    spec.push((
        ":specifiers_hash",
        params.specifiers_hash.unwrap().to_string(),
    ));
    spec
}

pub fn specifier_params<'a>(spec: &'a [(&'static str, String)]) -> Vec<(&'a str, &'a dyn ToSql)> {
    spec.iter()
        .map(|(k, v)| (k.as_ref(), v as &dyn ToSql))
        .collect_vec()
}
