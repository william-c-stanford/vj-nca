import unittest

from vj_nca.evaluate import recovery_frame


class EvaluationTests(unittest.TestCase):
    def test_static_incompetent_model_cannot_claim_recovery(self):
        self.assertIsNone(recovery_frame([0] * 20, [0.1] * 20, [0.1] * 20, False))

    def test_recovery_requires_sustained_agreement_and_target_accuracy(self):
        paired = [0.1, 0.05, 0.002, 0.001, 0.001, 0.001]
        self.assertEqual(recovery_frame(paired, [0.01] * 6, [0.01] * 6, True), 2)
        self.assertIsNone(recovery_frame(paired, [0.2] * 6, [0.01] * 6, True))
        self.assertIsNone(recovery_frame([0.1, 0.001, 0.1, 0.001], [0.01] * 4, [0.01] * 4, True))


if __name__ == "__main__":
    unittest.main()
