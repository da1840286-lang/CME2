import unittest

from cme.registry import SettingsError
from cme.smoke import frame_attempt_numbers, image_seed_for_attempt


class FrameAttemptDispatchTests(unittest.TestCase):
    def test_standalone_frame_preserves_workflow_retry_number(self):
        self.assertEqual(frame_attempt_numbers("frame", 2, 2), (2,))

    def test_standalone_frame_defaults_to_first_attempt(self):
        self.assertEqual(frame_attempt_numbers("frame", 1, 2), (1,))

    def test_all_stage_keeps_bounded_retry_sequence(self):
        self.assertEqual(frame_attempt_numbers("all", 2, 2), (1, 2))
        self.assertEqual(frame_attempt_numbers("all", 1, 1), (1,))

    def test_standalone_frame_rejects_nonpositive_attempt(self):
        with self.assertRaises(SettingsError):
            frame_attempt_numbers("frame", 0, 2)

    def test_image_seed_changes_between_controlled_attempts(self):
        contract_id = "contract-a844ddd31981a678"
        first = image_seed_for_attempt(0, 1, contract_id)
        second = image_seed_for_attempt(0, 2, contract_id)
        self.assertEqual(second, first + 1)


if __name__ == "__main__":
    unittest.main()
