use std::time::{Duration, SystemTime, UNIX_EPOCH};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

pub fn scaled_duration(duration: Duration, timescale: u64) -> u64 {
    (duration.as_nanos() * u128::from(timescale) / NANOS_PER_SECOND).min(u128::from(u64::MAX)) as u64
}

pub fn unscaled_duration(duration: u64, timescale: u64) -> Duration {
    let duration_nanos =
        (u128::from(duration) * NANOS_PER_SECOND / u128::from(timescale)).min(u128::from(u64::MAX)) as u64;
    Duration::from_nanos(duration_nanos)
}

pub fn mp4_timestamp(duration: Duration) -> u64 {
    duration.as_secs() + 2_082_844_800
}

pub fn mp4_timestamp_now() -> u64 {
    mp4_timestamp(now())
}

fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
}
