import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class ReleasedScoringTests(unittest.TestCase):
    def test_released_reverse_jsd_matches_upstream_literal(self):
        import torch
        from torch.nn import functional as functional

        from redeep_token.scoring import released_reverse_jsd

        post_ffn_logits = torch.tensor(
            [[1.2, -0.7, 0.3], [0.1, 0.2, 0.8]],
            dtype=torch.float32,
        )
        pre_ffn_logits = torch.tensor(
            [[0.2, 0.5, -0.1], [1.0, -0.3, 0.2]],
            dtype=torch.float32,
        )
        post_distribution = functional.softmax(post_ffn_logits, dim=-1)
        pre_distribution = functional.softmax(pre_ffn_logits, dim=-1)
        midpoint = 0.5 * (post_distribution + pre_distribution)
        expected = 0.5 * (
            functional.kl_div(
                functional.log_softmax(post_ffn_logits, dim=-1),
                midpoint,
                reduction="none",
            ).mean(-1)
            + functional.kl_div(
                functional.log_softmax(pre_ffn_logits, dim=-1),
                midpoint,
                reduction="none",
            ).mean(-1)
        ) * 1_000_000.0

        actual = released_reverse_jsd(post_ffn_logits, pre_ffn_logits)

        torch.testing.assert_close(actual, expected)

    def test_manual_top_context_ranking_matches_softmax_ranking(self):
        import torch

        from redeep_token.scoring import top_context_indices

        attention_logits = torch.tensor(
            [[0.2, 1.5, -0.1, 0.8], [2.0, 0.0, 1.0, -1.0]]
        )
        expected = torch.argsort(
            torch.softmax(attention_logits, dim=-1),
            dim=-1,
            descending=True,
        )[:, :2]

        actual = top_context_indices(attention_logits, top_fraction=0.5)

        torch.testing.assert_close(actual, expected)

    def test_qwen_projection_applies_q_and_k_norm_before_head_transpose(self):
        import torch

        from redeep_token.scoring import project_query_key

        class Scale(torch.nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor
                self.calls = 0

            def forward(self, values):
                self.calls += 1
                return values * self.factor

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4, bias=False)
                self.k_proj = torch.nn.Linear(4, 2, bias=False)
                self.q_norm = Scale(2.0)
                self.k_norm = Scale(3.0)
                with torch.no_grad():
                    self.q_proj.weight.copy_(torch.eye(4))
                    self.k_proj.weight.copy_(
                        torch.tensor(
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                            ]
                        )
                    )

        attention = Attention()
        hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

        query, key = project_query_key(
            attention,
            query_inputs=hidden,
            key_inputs=hidden,
            num_query_heads=2,
            num_key_value_heads=1,
            head_dim=2,
        )

        self.assertEqual(tuple(query.shape), (1, 2, 1, 2))
        self.assertEqual(tuple(key.shape), (1, 1, 1, 2))
        torch.testing.assert_close(
            query,
            torch.tensor([[[[2.0, 4.0]], [[6.0, 8.0]]]]),
        )
        torch.testing.assert_close(
            key,
            torch.tensor([[[[3.0, 6.0]]]]),
        )
        self.assertEqual(attention.q_norm.calls, 1)
        self.assertEqual(attention.k_norm.calls, 1)

    def test_tiny_qwen3_extractor_runs_the_complete_feature_path(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from redeep_token.scoring import ReDeEPFeatureExtractor

        torch.manual_seed(31)
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=101,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=64,
                attention_dropout=0.0,
                use_sliding_window=False,
                sliding_window=None,
            )
        ).eval()
        extractor = ReDeEPFeatureExtractor(
            model,
            candidate_heads=[(0, 0), (1, 3)],
            top_fraction=0.1,
            logit_chunk_size=2,
            cosine_chunk_size=2,
        )
        input_ids = [(index * 5 + 1) % 97 + 1 for index in range(28)]

        result = extractor.extract(
            input_ids=input_ids,
            prefix_length=20,
            score_positions=list(range(19, 27)),
        )

        self.assertEqual(len(result["external"]), 8)
        self.assertEqual(len(result["external"][0]), 2)
        self.assertEqual(len(result["parametric"]), 8)
        self.assertEqual(len(result["parametric"][0]), 2)
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(row)).all()
                for row in result["external"] + result["parametric"]
            )
        )

        with torch.inference_mode():
            reference = model(
                input_ids=torch.tensor([input_ids]),
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        final_hidden = reference.hidden_states[-1][0]
        score_positions = list(range(19, 27))
        expected_external = []
        for score_position in score_positions:
            row = []
            for layer_index, head_index in ((0, 0), (1, 3)):
                prefix_attention = reference.attentions[layer_index][
                    0,
                    head_index,
                    score_position,
                    :20,
                ]
                top_indices = torch.argsort(
                    prefix_attention,
                    descending=True,
                )[:2]
                attended = final_hidden[top_indices].mean(dim=0)
                row.append(
                    torch.nn.functional.cosine_similarity(
                        attended,
                        final_hidden[score_position],
                        dim=0,
                    )
                )
            expected_external.append(torch.stack(row))
        torch.testing.assert_close(
            torch.tensor(result["external"]),
            torch.stack(expected_external),
            atol=2e-6,
            rtol=2e-6,
        )


if __name__ == "__main__":
    unittest.main()
