use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
