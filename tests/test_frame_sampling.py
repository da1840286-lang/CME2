"""Adaptive, end-anchored frame sampling for Analyze.

Regression: a 21.48 s ad was sampled at 1.79 ... 19.69 s with the old midpoint scheme, so its closing result
shot (the last ~1.8 s) was never seen and the analysis mislabelled a close-up as 'area cleared'.
"""
import unittest

from cme.radar.analyze import (ADAPTIVE_MAX_FRAMES, ADAPTIVE_MIN_FRAMES, END_MARGIN_SEC, plan_frame_times)


class FrameSamplingTests(unittest.TestCase):
    def test_audited_21s_ad_now_samples_the_final_shot_and_the_hook(self):
        times, plan = plan_frame_times(21.478)
        self.assertEqual(times[0], 0.5)
        self.assertGreaterEqual(times[-1], 21.478 - END_MARGIN_SEC - 0.01)  # old scheme stopped at 19.69
        self.assertEqual(plan["mode"], "adaptive_anchored")
        self.assertLessEqual(plan["actual_spacing_sec"], 4.3)

    def test_old_midpoint_scheme_missed_the_ending(self):
        old = [round(21.478 * (i + 0.5) / 6, 2) for i in range(6)]
        self.assertEqual(old[-1], 19.69)
        self.assertLess(old[-1], 21.478 - 1.5)

    def test_frame_count_is_clamped_and_long_videos_are_flagged_as_capped(self):
        short, _ = plan_frame_times(6.0)
        self.assertEqual(len(short), ADAPTIVE_MIN_FRAMES)
        long, plan = plan_frame_times(53.2)
        self.assertEqual(len(long), ADAPTIVE_MAX_FRAMES)
        self.assertTrue(plan["capped"])
        self.assertGreater(plan["actual_spacing_sec"], 4.0)  # reported honestly, not hidden

    def test_times_are_increasing_unique_and_inside_the_video(self):
        for duration in (1.5, 3.0, 12.0, 21.478, 60.0):
            times, _ = plan_frame_times(duration)
            self.assertEqual(times, sorted(set(times)), duration)
            self.assertTrue(all(0 <= t < duration for t in times), duration)

    def test_degenerate_very_short_video_yields_one_midpoint_frame(self):
        self.assertEqual(plan_frame_times(0.8)[0], [0.4])

    def test_explicit_count_keeps_the_previous_behaviour(self):
        times, plan = plan_frame_times(6.0, count=3)
        self.assertEqual(times, [1.0, 3.0, 5.0])
        self.assertEqual(plan["mode"], "fixed_midpoints")


if __name__ == "__main__":
    unittest.main()
