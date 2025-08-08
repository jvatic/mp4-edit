use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub fn scaled_duration(duration: Duration, timescale: u64) -> u64 {
    const NANOS_PER_SECOND: u128 = 1_000_000_000;
    let duration_nanos = duration.as_nanos();
    let timescale = timescale as u128;
    (duration_nanos * timescale / NANOS_PER_SECOND) as u64
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
