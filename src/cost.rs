#[derive(Clone, Copy, PartialEq)]
pub struct Cost {
    pub efficiency: f64,
    pub safety: f64,
    pub accel: f64,
    pub steer: f64,

    pub discount: f64,
    pub discount_factor: f64,

    pub weight: f64,
}

impl std::fmt::Display for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.normalize();
        write_f!(
            f,
            "{s.efficiency:8.2} {s.safety:8.2} {s.accel:8.2} {s.steer:8.2}"
        )
    }
}

impl std::fmt::Debug for Cost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self;
        write_f!(
            f,
            "eff: {s.efficiency:.2}, safe: {s.safety:.2}, accel: {s.accel:.2}, steer: {s.steer:.2}"
        )
    }
}

impl Cost {
    pub const ZERO: Self = Self::new(1.0, 1.0);

    pub const fn new(discount_factor: f64, weight: f64) -> Self {
        Self {
            efficiency: 0.0,
            safety: 0.0,
            accel: 0.0,
            steer: 0.0,
            discount: 1.0,
            discount_factor,
            weight,
        }
    }

    pub fn max_value() -> Self {
        Self {
            efficiency: f64::MAX,
            safety: 0.0,
            accel: 0.0,
            steer: 0.0,
            discount: 1.0,
            discount_factor: 1.0,
            weight: 1.0,
        }
    }

    pub fn normalize(&self) -> Self {
        Self {
            efficiency: self.efficiency * self.weight,
            safety: self.safety * self.weight,
            accel: self.accel * self.weight,
            steer: self.steer * self.weight,
            discount: 1.0,
            discount_factor: 1.0,
            weight: 1.0,
        }
    }

    fn unweighted_total(&self) -> f64 {
        self.efficiency + self.safety + self.accel + self.steer
    }

    pub fn total(&self) -> f64 {
        self.weight * self.unweighted_total()
    }

    pub fn update_discount(&mut self, dt: f64) {
        self.discount *= self.discount_factor.powf(dt);
    }

    pub fn max(&self, other: &Self) -> Self {
        if self > other {
            *self
        } else {
            *other
        }
    }
}

impl Default for Cost {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.total().partial_cmp(&other.total())
    }
}

impl std::iter::Sum for Cost {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Cost::new(1.0, 1.0);
        for r in iter {
            sum += r;
        }
        sum
    }
}

impl std::ops::Mul<f64> for Cost {
    type Output = Cost;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            efficiency: self.efficiency * rhs,
            safety: self.safety * rhs,
            accel: self.accel * rhs,
            steer: self.steer * rhs,
            discount: self.discount,
            discount_factor: self.discount_factor,
            weight: self.weight,
        }
    }
}

impl std::ops::Div<f64> for Cost {
    type Output = Cost;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            efficiency: self.efficiency / rhs,
            safety: self.safety / rhs,
            accel: self.accel / rhs,
            steer: self.steer / rhs,
            discount: self.discount,
            discount_factor: self.discount_factor,
            weight: self.weight,
        }
    }
}

impl std::ops::DivAssign<f64> for Cost {
    fn div_assign(&mut self, rhs: f64) {
        self.efficiency /= rhs;
        self.safety /= rhs;
        self.accel /= rhs;
        self.steer /= rhs;
    }
}

impl std::ops::Add for Cost {
    type Output = Cost;

    fn add(self, rhs: Self) -> Self::Output {
        let a = self.normalize();
        let b = rhs.normalize();
        Self {
            efficiency: a.efficiency + b.efficiency,
            safety: a.safety + b.safety,
            accel: a.accel + b.accel,
            steer: a.steer + b.steer,
            discount: self.discount,
            discount_factor: self.discount_factor,
            weight: 1.0,
        }
    }
}

impl std::ops::Sub for Cost {
    type Output = Cost;

    fn sub(self, rhs: Self) -> Self::Output {
        let a = self.normalize();
        let b = rhs.normalize();
        Self {
            efficiency: a.efficiency - b.efficiency,
            safety: a.safety - b.safety,
            accel: a.accel - b.accel,
            steer: a.steer - b.steer,
            discount: self.discount,
            discount_factor: self.discount_factor,
            weight: 1.0,
        }
    }
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
