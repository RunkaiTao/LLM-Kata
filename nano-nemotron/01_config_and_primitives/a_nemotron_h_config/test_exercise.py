import dataclasses
import pytest
from exercise import NemotronHConfig


class TestNemotronHConfigIsDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(NemotronHConfig)


class TestGeneralDefaults:
    def test_vocab_size(self):
        config = NemotronHConfig()
        assert config.vocab_size == 131072

    def test_hidden_size(self):
        config = NemotronHConfig()
        assert config.hidden_size == 2688

    def test_intermediate_size(self):
        config = NemotronHConfig()
        assert config.intermediate_size == 1856

    def test_num_hidden_layers(self):
        config = NemotronHConfig()
        assert config.num_hidden_layers == 52

    def test_layer_norm_epsilon(self):
        config = NemotronHConfig()
        assert config.layer_norm_epsilon == 1e-5

    def test_tie_word_embeddings(self):
        config = NemotronHConfig()
        assert config.tie_word_embeddings is False


class TestHybridPattern:
    def test_default_pattern(self):
        config = NemotronHConfig()
        assert config.hybrid_override_pattern == "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

    def test_pattern_length_matches_layers(self):
        config = NemotronHConfig()
        assert len(config.hybrid_override_pattern) == config.num_hidden_layers

    def test_pattern_only_valid_chars(self):
        config = NemotronHConfig()
        for ch in config.hybrid_override_pattern:
            assert ch in {"M", "*", "-", "E"}, f"Invalid character '{ch}' in pattern"


class TestAttentionDefaults:
    def test_num_attention_heads(self):
        config = NemotronHConfig()
        assert config.num_attention_heads == 32

    def test_head_dim(self):
        config = NemotronHConfig()
        assert config.head_dim == 128

    def test_num_key_value_heads(self):
        config = NemotronHConfig()
        assert config.num_key_value_heads == 2

    def test_gqa_ratio(self):
        """GQA: num_attention_heads should be a multiple of num_key_value_heads."""
        config = NemotronHConfig()
        assert config.num_attention_heads % config.num_key_value_heads == 0

    def test_attention_bias(self):
        config = NemotronHConfig()
        assert config.attention_bias is False


class TestMambaDefaults:
    def test_mamba_num_heads(self):
        config = NemotronHConfig()
        assert config.mamba_num_heads == 64

    def test_mamba_head_dim(self):
        config = NemotronHConfig()
        assert config.mamba_head_dim == 64

    def test_mamba_n_groups(self):
        config = NemotronHConfig()
        assert config.mamba_n_groups == 8

    def test_ssm_state_size(self):
        config = NemotronHConfig()
        assert config.ssm_state_size == 128

    def test_mamba_d_conv(self):
        config = NemotronHConfig()
        assert config.mamba_d_conv == 4

    def test_mamba_expand(self):
        config = NemotronHConfig()
        assert config.mamba_expand == 2

    def test_mamba_intermediate_size(self):
        """Mamba intermediate size is num_heads * head_dim."""
        config = NemotronHConfig()
        mamba_intermediate = config.mamba_num_heads * config.mamba_head_dim
        assert mamba_intermediate == 4096


class TestMoEDefaults:
    def test_n_routed_experts(self):
        config = NemotronHConfig()
        assert config.n_routed_experts == 128

    def test_n_shared_experts(self):
        config = NemotronHConfig()
        assert config.n_shared_experts == 1

    def test_moe_intermediate_size(self):
        config = NemotronHConfig()
        assert config.moe_intermediate_size == 1856

    def test_num_experts_per_tok(self):
        config = NemotronHConfig()
        assert config.num_experts_per_tok == 6

    def test_routed_scaling_factor(self):
        config = NemotronHConfig()
        assert config.routed_scaling_factor == 2.5

    def test_moe_shared_expert_intermediate_size(self):
        config = NemotronHConfig()
        assert config.moe_shared_expert_intermediate_size == 3712


class TestAttentionExtras:
    def test_max_position_embeddings(self):
        config = NemotronHConfig()
        assert config.max_position_embeddings == 262144

    def test_rope_theta(self):
        config = NemotronHConfig()
        assert config.rope_theta == 10000.0


class TestLayersBlockType:
    def test_returns_list(self):
        config = NemotronHConfig()
        result = config.layers_block_type
        assert isinstance(result, list)

    def test_length_matches_layers(self):
        config = NemotronHConfig()
        assert len(config.layers_block_type) == config.num_hidden_layers

    def test_correct_mapping(self):
        config = NemotronHConfig(
            num_hidden_layers=4,
            hybrid_override_pattern="M*-E",
        )
        assert config.layers_block_type == ["mamba", "attention", "mlp", "moe"]

    def test_all_mamba_and_mlp_pattern(self):
        config = NemotronHConfig(
            num_hidden_layers=3,
            hybrid_override_pattern="M-M",
        )
        assert config.layers_block_type == ["mamba", "mlp", "mamba"]


class TestCustomValues:
    def test_override_all(self):
        config = NemotronHConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=4,
            hybrid_override_pattern="M-M*",
            num_attention_heads=4,
            head_dim=16,
            num_key_value_heads=2,
        )
        assert config.vocab_size == 256
        assert config.hidden_size == 64
        assert config.intermediate_size == 256
        assert config.num_hidden_layers == 4
        assert config.hybrid_override_pattern == "M-M*"
        assert config.num_attention_heads == 4
        assert config.head_dim == 16
        assert config.num_key_value_heads == 2
