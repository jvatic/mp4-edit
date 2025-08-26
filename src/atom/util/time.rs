use std::{
    fmt::Debug,
    ops::RangeBounds,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

pub fn scaled_duration(duration: Duration, timescale: u64) -> u64 {
    (duration.as_nanos() * u128::from(timescale) / NANOS_PER_SECOND).min(u128::from(u64::MAX))
        as u64
}

pub fn unscaled_duration(duration: u64, timescale: u64) -> Duration {
    let duration_nanos = (u128::from(duration) * NANOS_PER_SECOND / u128::from(timescale))
        .min(u128::from(u64::MAX)) as u64;
    Duration::from_nanos(duration_nanos)
}

pub fn scaled_duration_range(
    range: impl RangeBounds<Duration>,
    timescale: u64,
) -> impl RangeBounds<u64> + Debug {
    use std::ops::Bound;
    let start = match range.start_bound() {
        Bound::Included(start) => Bound::Included(scaled_duration(*start, timescale)),
        Bound::Excluded(start) => Bound::Excluded(scaled_duration(*start, timescale)),
        Bound::Unbounded => Bound::Unbounded,
    };
    let end = match range.end_bound() {
        Bound::Included(end) => Bound::Included(scaled_duration(*end, timescale)),
        Bound::Excluded(end) => Bound::Excluded(scaled_duration(*end, timescale)),
        Bound::Unbounded => Bound::Unbounded,
    };
    (start, end)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Bound;
    use std::time::Duration;

    fn test_scaled_duration_range<F>(test_case: F)
    where
        F: FnOnce() -> ScaledDurationRangeTestCase,
    {
        let test_case = test_case();

        let range = (test_case.start_bound, test_case.end_bound);
        let scaled_range = scaled_duration_range(range, test_case.timescale);

        match (&test_case.expected_start, scaled_range.start_bound()) {
            (Bound::Included(expected), Bound::Included(actual)) => {
                assert_eq!(
                    *expected, *actual,
                    "Start bound mismatch in {}",
                    test_case.description
                )
            }
            (Bound::Excluded(expected), Bound::Excluded(actual)) => {
                assert_eq!(
                    *expected, *actual,
                    "Start bound mismatch in {}",
                    test_case.description
                )
            }
            (Bound::Unbounded, Bound::Unbounded) => (),
            _ => panic!("Start bound type mismatch in {}", test_case.description),
        }

        match (&test_case.expected_end, scaled_range.end_bound()) {
            (Bound::Included(expected), Bound::Included(actual)) => {
                assert_eq!(
                    *expected, *actual,
                    "End bound mismatch in {}",
                    test_case.description
                )
            }
            (Bound::Excluded(expected), Bound::Excluded(actual)) => {
                assert_eq!(
                    *expected, *actual,
                    "End bound mismatch in {}",
                    test_case.description
                )
            }
            (Bound::Unbounded, Bound::Unbounded) => (),
            _ => panic!("End bound type mismatch in {}", test_case.description),
        }
    }

    struct ScaledDurationRangeTestCase {
        start_bound: Bound<Duration>,
        end_bound: Bound<Duration>,
        timescale: u64,
        expected_start: Bound<u64>,
        expected_end: Bound<u64>,
        description: &'static str,
    }

    macro_rules! test_scaled_duration_range {
        ($($name:ident => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    test_scaled_duration_range($test_case);
                }
            )*
        };
    }

    test_scaled_duration_range!(
        included_included => || ScaledDurationRangeTestCase {
            start_bound: Bound::Included(Duration::from_secs(1)),
            end_bound: Bound::Included(Duration::from_secs(3)),
            timescale: 1000,
            expected_start: Bound::Included(1000),
            expected_end: Bound::Included(3000),
            description: "included/included",
        },
        included_excluded => || ScaledDurationRangeTestCase {
            start_bound: Bound::Included(Duration::from_secs(1)),
            end_bound: Bound::Excluded(Duration::from_secs(4)),
            timescale: 44100,
            expected_start: Bound::Included(44100),
            expected_end: Bound::Excluded(176400),
            description: "included/excluded",
        },
        included_unbounded => || ScaledDurationRangeTestCase {
            start_bound: Bound::Included(Duration::from_millis(500)),
            end_bound: Bound::Unbounded,
            timescale: 48000,
            expected_start: Bound::Included(24000),
            expected_end: Bound::Unbounded,
            description: "included/unbounded",
        },
        excluded_included => || ScaledDurationRangeTestCase {
            start_bound: Bound::Excluded(Duration::from_millis(100)),
            end_bound: Bound::Included(Duration::from_secs(2)),
            timescale: 48000,
            expected_start: Bound::Excluded(4800),
            expected_end: Bound::Included(96000),
            description: "excluded/included",
        },
        excluded_excluded => || ScaledDurationRangeTestCase {
            start_bound: Bound::Excluded(Duration::from_secs(1)),
            end_bound: Bound::Excluded(Duration::from_secs(3)),
            timescale: 1000,
            expected_start: Bound::Excluded(1000),
            expected_end: Bound::Excluded(3000),
            description: "excluded/excluded",
        },
        excluded_unbounded => || ScaledDurationRangeTestCase {
            start_bound: Bound::Excluded(Duration::from_millis(250)),
            end_bound: Bound::Unbounded,
            timescale: 8000,
            expected_start: Bound::Excluded(2000),
            expected_end: Bound::Unbounded,
            description: "excluded/unbounded",
        },
        unbounded_included => || ScaledDurationRangeTestCase {
            start_bound: Bound::Unbounded,
            end_bound: Bound::Included(Duration::from_secs(5)),
            timescale: 1000,
            expected_start: Bound::Unbounded,
            expected_end: Bound::Included(5000),
            description: "unbounded/included",
        },
        unbounded_excluded => || ScaledDurationRangeTestCase {
            start_bound: Bound::Unbounded,
            end_bound: Bound::Excluded(Duration::from_secs(3)),
            timescale: 22050,
            expected_start: Bound::Unbounded,
            expected_end: Bound::Excluded(66150),
            description: "unbounded/excluded",
        },
        unbounded_unbounded => || ScaledDurationRangeTestCase {
            start_bound: Bound::Unbounded,
            end_bound: Bound::Unbounded,
            timescale: 1000,
            expected_start: Bound::Unbounded,
            expected_end: Bound::Unbounded,
            description: "unbounded/unbounded",
        },
        zero_timescale => || ScaledDurationRangeTestCase {
            start_bound: Bound::Included(Duration::from_secs(1)),
            end_bound: Bound::Included(Duration::from_secs(2)),
            timescale: 0,
            expected_start: Bound::Included(0),
            expected_end: Bound::Included(0),
            description: "zero timescale",
        },
        large_timescale => || ScaledDurationRangeTestCase {
            start_bound: Bound::Included(Duration::from_millis(1)),
            end_bound: Bound::Included(Duration::from_millis(2)),
            timescale: 1_000_000,
            expected_start: Bound::Included(1000),
            expected_end: Bound::Included(2000),
            description: "large timescale",
        },
        small_timescale => || ScaledDurationRangeTestCase {
            start_bound: Bound::Included(Duration::from_secs(1)),
            end_bound: Bound::Included(Duration::from_secs(2)),
            timescale: 1,
            expected_start: Bound::Included(1),
            expected_end: Bound::Included(2),
            description: "small timescale",
        },
        overflow_protection => || ScaledDurationRangeTestCase {
            start_bound: Bound::Included(Duration::from_secs(u64::MAX)),
            end_bound: Bound::Included(Duration::from_secs(u64::MAX)),
            timescale: 1000,
            expected_start: Bound::Included(u64::MAX),
            expected_end: Bound::Included(u64::MAX),
            description: "overflow protection",
        },
    );
}
