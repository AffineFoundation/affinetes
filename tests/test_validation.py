"""Tests for input validation utilities."""

import pytest

from affinetes.utils.validation import (
    validate_image_tag,
    validate_replicas,
    validate_env_vars,
    validate_mem_limit,
    validate_load_balance_strategy,
    validate_hosts,
)
from affinetes.utils.exceptions import ValidationError


class TestValidateImageTag:
    """Tests for validate_image_tag."""

    def test_simple_tag(self):
        assert validate_image_tag("myimage:latest") == "myimage:latest"

    def test_registry_tag(self):
        assert validate_image_tag("docker.io/user/repo:v1") == "docker.io/user/repo:v1"

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError):
            validate_image_tag("")

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_image_tag(None)

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            validate_image_tag("   ")

    def test_strips_whitespace(self):
        assert validate_image_tag("  myimage:v1  ") == "myimage:v1"

    def test_invalid_chars_raises(self):
        with pytest.raises(ValidationError):
            validate_image_tag("image name with spaces")

    def test_starts_with_special_raises(self):
        with pytest.raises(ValidationError):
            validate_image_tag("-myimage")


class TestValidateReplicas:
    """Tests for validate_replicas."""

    def test_valid_replicas(self):
        assert validate_replicas(1) == 1
        assert validate_replicas(10) == 10

    def test_zero_raises(self):
        with pytest.raises(ValidationError):
            validate_replicas(0)

    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            validate_replicas(-1)

    def test_float_raises(self):
        with pytest.raises(ValidationError):
            validate_replicas(1.5)

    def test_string_raises(self):
        with pytest.raises(ValidationError):
            validate_replicas("3")


class TestValidateEnvVars:
    """Tests for validate_env_vars."""

    def test_none_returns_none(self):
        assert validate_env_vars(None) is None

    def test_valid_dict(self):
        env = {"KEY": "value", "FOO": "bar"}
        assert validate_env_vars(env) == env

    def test_non_dict_raises(self):
        with pytest.raises(ValidationError, match="must be a dict"):
            validate_env_vars([("KEY", "val")])

    def test_empty_key_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            validate_env_vars({"": "value"})

    def test_non_string_value_raises(self):
        with pytest.raises(ValidationError, match="must be strings"):
            validate_env_vars({"KEY": 123})

    def test_int_key_raises(self):
        with pytest.raises(ValidationError):
            validate_env_vars({42: "value"})


class TestValidateMemLimit:
    """Tests for validate_mem_limit."""

    def test_none_returns_none(self):
        assert validate_mem_limit(None) is None

    def test_valid_formats(self):
        assert validate_mem_limit("512m") == "512m"
        assert validate_mem_limit("1g") == "1g"
        assert validate_mem_limit("2Gi") == "2Gi"
        assert validate_mem_limit("256Mi") == "256Mi"

    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError):
            validate_mem_limit("lots")

    def test_non_string_raises(self):
        with pytest.raises(ValidationError):
            validate_mem_limit(512)


class TestValidateLoadBalanceStrategy:
    """Tests for validate_load_balance_strategy."""

    def test_valid_strategies(self):
        assert validate_load_balance_strategy("random") == "random"
        assert validate_load_balance_strategy("round_robin") == "round_robin"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValidationError, match="Invalid load_balance"):
            validate_load_balance_strategy("weighted")


class TestValidateHosts:
    """Tests for validate_hosts."""

    def test_none_returns_none(self):
        assert validate_hosts(None, 3) is None

    def test_valid_hosts(self):
        hosts = ["host1", "host2", "host3"]
        assert validate_hosts(hosts, 3) == hosts

    def test_insufficient_hosts_raises(self):
        with pytest.raises(ValidationError, match="Not enough hosts"):
            validate_hosts(["h1"], 3)

    def test_non_list_raises(self):
        with pytest.raises(ValidationError, match="must be a list"):
            validate_hosts("host1", 1)

    def test_empty_host_string_raises(self):
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_hosts(["host1", ""], 2)

    def test_non_string_host_raises(self):
        with pytest.raises(ValidationError, match="non-empty string"):
            validate_hosts([123], 1)
