use core::fmt::Debug;
use core::ops::AddAssign;
use num_traits::{cast::FromPrimitive, float::Float, identities::One, identities::Zero};
use rolling_stats::Stats;

#[derive(Clone)]
pub struct CostSet<
    F: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug = f64,
    T = (),
> {
    costs: Vec<(F, T)>,
    stats: Stats<F>,
}

impl std::fmt::Debug for CostSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostSet")
            .field("costs", &self.costs)
            .field("mean", &self.stats.mean)
            .field("std_dev", &self.stats.std_dev)
            .finish()
    }
}

impl<F: Float + Zero + One + AddAssign + FromPrimitive + PartialEq + Debug, T: Clone + Default>
    CostSet<F, T>
{
    pub fn new() -> Self {
        Self {
            costs: Vec::new(),
            stats: Stats::new(),
        }
    }

    pub fn push(&mut self, cost: (F, T)) {
        let cost_val = cost.0;

        self.costs.push(cost);
        self.stats.update(cost_val);
    }

    pub fn mean(&self) -> F {
        self.stats.mean
    }

    pub fn std_dev(&self) -> F {
        if self.stats.std_dev.is_finite() {
            self.stats.std_dev
        } else {
            F::from_f64(1e12).unwrap()
        }
    }

    pub fn len(&self) -> usize {
        self.costs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.costs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(F, T)> {
        self.costs.iter()
    }
}
