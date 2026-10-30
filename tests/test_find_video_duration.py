"""Regression tests for the FIND-only video-duration selector."""
import unittest

from cme.radar.find import VIDEO_DURATION_RANGES, _video_duration_filter_reason


class VideoDurationFilterTests(unittest.TestCase):
    def test_default_range_keeps_known_and_unknown_durations(self):
        self.assertEqual(VIDEO_DURATION_RANGES[0], "all")
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": None}, "all"))
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 64.6}, "all"))

    def test_up_to_40_seconds_includes_exact_boundary(self):
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 40}, "up_to_40"))
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 12.5}, "up_to_40"))
        self.assertEqual(_video_duration_filter_reason({"duration_seconds": 40.01}, "up_to_40"), "VIDEO_DURATION_OUT_OF_RANGE")

    def test_40_to_60_seconds_has_non_overlapping_boundaries(self):
        self.assertEqual(_video_duration_filter_reason({"duration_seconds": 40}, "40_to_60"), "VIDEO_DURATION_OUT_OF_RANGE")
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 40.01}, "40_to_60"))
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 60}, "40_to_60"))
        self.assertEqual(_video_duration_filter_reason({"duration_seconds": 60.01}, "40_to_60"), "VIDEO_DURATION_OUT_OF_RANGE")

    def test_60_to_90_seconds_has_non_overlapping_boundaries(self):
        self.assertEqual(_video_duration_filter_reason({"duration_seconds": 60}, "60_to_90"), "VIDEO_DURATION_OUT_OF_RANGE")
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 60.01}, "60_to_90"))
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 90}, "60_to_90"))
        self.assertEqual(_video_duration_filter_reason({"duration_seconds": 90.01}, "60_to_90"), "VIDEO_DURATION_OUT_OF_RANGE")

    def test_over_90_seconds(self):
        self.assertEqual(_video_duration_filter_reason({"duration_seconds": 90}, "over_90"), "VIDEO_DURATION_OUT_OF_RANGE")
        self.assertIsNone(_video_duration_filter_reason({"duration_seconds": 90.01}, "over_90"))

    def test_unknown_invalid_and_non_finite_durations_are_excluded_from_selected_ranges(self):
        for value in (None, "", "unknown", 0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.assertEqual(_video_duration_filter_reason({"duration_seconds": value}, "up_to_40"), "VIDEO_DURATION_UNKNOWN")

    def test_unknown_selection_name_is_rejected(self):
        with self.assertRaises(ValueError):
            _video_duration_filter_reason({"duration_seconds": 20}, "forty-ish")


if __name__ == "__main__":
    unittest.main()
