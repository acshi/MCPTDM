use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct RateTimer {
    interval: Duration,
    last: Instant,
}

impl RateTimer {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last: Instant::now(),
        }
    }

    #[allow(unused)]
    pub fn from_millis(interval_ms: u64) -> Self {
        Self::new(Duration::from_millis(interval_ms))
    }

    pub fn wait_until_ready(&mut self) {
        let now = Instant::now();
        let passed = now - self.last;

        if passed < self.interval {
            let wait_time = self.interval - passed;
            // eprintln!(
            //     "Doing a wait for {:.2}ms\n",
            //     wait_time.as_secs_f32() * 1000.0
            // );
            std::thread::sleep(wait_time);
        }

        assert!(self.ready());
    }

    pub fn ready(&mut self) -> bool {
        let now = Instant::now();
        let ready = (now - self.last) >= self.interval;

        if ready {
            self.last += self.interval;
            if self.last < now {
                self.last = now;
            }
        }

        ready
    }
}
