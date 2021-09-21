// This file based on SMPyBandits/SMPyBandits/Policies/kullback.py from https://github.com/SMPyBandits/SMPyBandits

fn kl_diverg(p: f64, q: f64) -> f64 {
    if p == 0.0 && q == 0.0 || p == 1.0 && q == 1.0 {
        return 0.0;
    }
    if q == 0.0 || q == 1.0 {
        return f64::INFINITY;
    }

    p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln()
}

fn klucb_gauss(x: f64, max_divergence: f64, sig2x: f64) -> f64 {
    x + (2.0 * sig2x * max_divergence).abs().sqrt()
}

// midpoint search optimization
fn klucb(x: f64, max_divergence: f64, high: f64, precision: f64, max_iters: usize) -> f64 {
    let mut value = x;
    let mut upper = high;
    for k in 0.. {
        if k > max_iters || upper - value < precision {
            break;
        }

        let mid = (value + upper) * 0.5;
        if kl_diverg(x, mid) > max_divergence {
            upper = mid;
        } else {
            value = mid;
        }
    }

    (value + upper) * 0.5
}

pub fn klucb_bernoulli(x: f64, max_divergence: f64) -> f64 {
    let upper = klucb_gauss(x, max_divergence, 0.25).min(1.0);
    klucb(x, max_divergence, upper, 1e-6, 50)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_klucb_bernoulli() {
        assert_abs_diff_eq!(klucb_bernoulli(0.1, 0.2), 0.378391, epsilon = 1e-6);
        assert_abs_diff_eq!(klucb_bernoulli(0.5, 0.2), 0.787088, epsilon = 1e-6);
        assert_abs_diff_eq!(klucb_bernoulli(0.9, 0.2), 0.994489, epsilon = 1e-6);

        assert_abs_diff_eq!(klucb_bernoulli(0.1, 0.4), 0.519475, epsilon = 1e-6);
        assert_abs_diff_eq!(klucb_bernoulli(0.1, 0.9), 0.734714, epsilon = 1e-6);

        assert_abs_diff_eq!(klucb_bernoulli(1.0, 0.0), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(klucb_bernoulli(1.0, 0.5), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(klucb_bernoulli(1.0, 1.0), 1.0, epsilon = 1e-6);
    }
}
