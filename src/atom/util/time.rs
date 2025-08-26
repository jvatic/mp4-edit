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

pub fn duration_sub_range(duration: Duration, range: impl RangeBounds<Duration>) -> Duration {
    use std::ops::Bound;
    let start = match range.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => *start + Duration::from_nanos(1),
        Bound::Unbounded => Duration::ZERO,
    };
    let end = match range.end_bound() {
        Bound::Included(end) => *end,
        Bound::Excluded(end) => *end - Duration::from_nanos(1),
        Bound::Unbounded => duration,
    };
    let range_duration = end.saturating_sub(start);
    duration.saturating_sub(range_duration)
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

    fn test_duration_sub_range<F>(test_case: F)
    where
        F: FnOnce() -> DurationSubRangeTestCase,
    {
        let test_case = test_case();

        let range = (test_case.start_bound, test_case.end_bound);
        let actual = duration_sub_range(test_case.input_duration, range);
        assert_eq!(
            actual, test_case.expected,
            "Result mismatch in {}",
            test_case.description
        );
    }

    struct DurationSubRangeTestCase {
        input_duration: Duration,
        start_bound: Bound<Duration>,
        end_bound: Bound<Duration>,
        expected: Duration,
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

    macro_rules! test_duration_sub_range {
        ($($name:ident => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    test_duration_sub_range($test_case);
                }
            )*
        };
    }

    test_duration_sub_range!(
        duration_sub_included_included => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(2)),
            end_bound: Bound::Included(Duration::from_secs(7)),
            expected: Duration::from_secs(5), // 10s - (7s - 2s) = 5s
            description: "included/included",
        },
        duration_sub_included_excluded => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(15),
            start_bound: Bound::Included(Duration::from_secs(2)),
            end_bound: Bound::Excluded(Duration::from_secs(9)),
            expected: Duration::from_secs(8) + Duration::from_nanos(1), // 15s - (7s - 1ns) = 8s + 1ns
            description: "included/excluded",
        },
        duration_sub_included_unbounded => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(3)),
            end_bound: Bound::Unbounded,
            expected: Duration::from_secs(3), // 10s - (10s - 3s) = 3s
            description: "included/unbounded",
        },
        duration_sub_excluded_included => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Excluded(Duration::from_secs(3)),
            end_bound: Bound::Included(Duration::from_secs(8)),
            expected: Duration::from_secs(5) + Duration::from_nanos(1), // 10s - (8s - 3s - 1ns) = 5s + 1ns
            description: "excluded/included",
        },
        duration_sub_excluded_excluded => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Excluded(Duration::from_secs(2)),
            end_bound: Bound::Excluded(Duration::from_secs(7)),
            expected: Duration::from_secs(5) + Duration::from_nanos(2), // 10s - (5s - 2ns) = 5s + 2ns
            description: "excluded/excluded",
        },
        duration_sub_excluded_unbounded => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(12),
            start_bound: Bound::Excluded(Duration::from_secs(4)),
            end_bound: Bound::Unbounded,
            expected: Duration::from_secs(4) + Duration::from_nanos(1), // 12s - (8s - 1ns) = 4s + 1ns
            description: "excluded/unbounded",
        },
        duration_sub_unbounded_included => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Unbounded,
            end_bound: Bound::Included(Duration::from_secs(5)),
            expected: Duration::from_secs(5), // 10s - (5s - 0s) = 5s
            description: "unbounded/included",
        },
        duration_sub_unbounded_excluded => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(20),
            start_bound: Bound::Unbounded,
            end_bound: Bound::Excluded(Duration::from_secs(6)),
            expected: Duration::from_secs(14) + Duration::from_nanos(1), // 20s - (6s - 1ns) = 14s + 1ns
            description: "unbounded/excluded",
        },
        duration_sub_unbounded_unbounded => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Unbounded,
            end_bound: Bound::Unbounded,
            expected: Duration::ZERO, // 10s - 10s = 0s
            description: "unbounded/unbounded",
        },
        duration_sub_end_before_start => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(7)),
            end_bound: Bound::Included(Duration::from_secs(3)),
            expected: Duration::from_secs(10), // 10s - 0s = 10s (since end < start, range duration = 0)
            description: "end before start",
        },
        duration_sub_start_beyond_total => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(5),
            start_bound: Bound::Included(Duration::from_secs(7)),
            end_bound: Bound::Included(Duration::from_secs(10)),
            expected: Duration::from_secs(2), // 5s - (10s - 7s) = 5s - 3s = 2s
            description: "start beyond total",
        },
        duration_sub_end_beyond_total => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(5),
            start_bound: Bound::Included(Duration::from_secs(2)),
            end_bound: Bound::Included(Duration::from_secs(10)),
            expected: Duration::ZERO, // 5s - 8s = 0s (saturating_sub)
            description: "end beyond total",
        },
        duration_sub_zero_total_duration => || DurationSubRangeTestCase {
            input_duration: Duration::ZERO,
            start_bound: Bound::Included(Duration::from_secs(1)),
            end_bound: Bound::Included(Duration::from_secs(2)),
            expected: Duration::ZERO, // 0s - 1s = 0s (saturating_sub)
            description: "zero total duration",
        },
        duration_sub_same_start_and_end => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(5)),
            end_bound: Bound::Included(Duration::from_secs(5)),
            expected: Duration::from_secs(10), // 10s - 0s = 10s (since start == end, range duration = 0)
            description: "same start and end",
        },
        duration_sub_nanosecond_precision => || DurationSubRangeTestCase {
            input_duration: Duration::from_nanos(1000),
            start_bound: Bound::Included(Duration::from_nanos(100)),
            end_bound: Bound::Included(Duration::from_nanos(900)),
            expected: Duration::from_nanos(200), // 1000ns - 800ns = 200ns
            description: "nanosecond precision",
        },
        duration_sub_excluded_bounds_edge_case => || DurationSubRangeTestCase {
            input_duration: Duration::from_nanos(10),
            start_bound: Bound::Excluded(Duration::from_nanos(3)),
            end_bound: Bound::Excluded(Duration::from_nanos(7)),
            expected: Duration::from_nanos(8), // 10ns - 2ns = 8ns
            description: "excluded bounds edge case",
        },
        duration_sub_excluded_same_bounds => || DurationSubRangeTestCase {
            input_duration: Duration::from_secs(10),
            start_bound: Bound::Excluded(Duration::from_secs(5)),
            end_bound: Bound::Excluded(Duration::from_secs(5)),
            expected: Duration::from_secs(10), // 10s - 0s = 10s (since excluded same bounds = 0 duration)
            description: "excluded same bounds",
        },
        duration_sub_minimal_excluded_range => || DurationSubRangeTestCase {
            input_duration: Duration::from_nanos(100),
            start_bound: Bound::Excluded(Duration::from_nanos(50)),
            end_bound: Bound::Excluded(Duration::from_nanos(52)),
            expected: Duration::from_nanos(100), // 100ns - 0ns = 100ns (since range duration = 0)
            description: "minimal excluded range",
        },
    );
}
