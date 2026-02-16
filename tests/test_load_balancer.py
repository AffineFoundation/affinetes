"""Tests for load balancer strategies and instance selection."""

import pytest
from unittest.mock import MagicMock

from affinetes.core.load_balancer import LoadBalancer, InstanceInfo
from affinetes.utils.exceptions import BackendError


class TestInstanceInfo:
    """Tests for InstanceInfo dataclass."""

    def test_instance_info_creation(self):
        """Test creating an InstanceInfo with default values."""
        backend = MagicMock()
        info = InstanceInfo(host="localhost", port=8000, backend=backend)
        assert info.host == "localhost"
        assert info.port == 8000
        assert info.request_count == 0

    def test_instance_info_str(self):
        """Test string representation of InstanceInfo."""
        backend = MagicMock()
        info = InstanceInfo(host="192.168.1.1", port=9000, backend=backend)
        assert str(info) == "192.168.1.1:9000"

    def test_instance_info_custom_request_count(self):
        """Test creating InstanceInfo with custom request count."""
        backend = MagicMock()
        info = InstanceInfo(host="host1", port=8000, backend=backend, request_count=42)
        assert info.request_count == 42


class TestLoadBalancer:
    """Tests for LoadBalancer class."""

    def test_init_random_strategy(self):
        """Test initialization with random strategy."""
        lb = LoadBalancer(strategy="random")
        assert lb._strategy == "random"

    def test_init_round_robin_strategy(self):
        """Test initialization with round_robin strategy."""
        lb = LoadBalancer(strategy="round_robin")
        assert lb._strategy == "round_robin"

    def test_init_invalid_strategy_raises(self):
        """Test initialization with invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            LoadBalancer(strategy="least_connections")

    def test_select_instance_empty_list_raises(self):
        """Test selecting from empty instance list raises BackendError."""
        lb = LoadBalancer(strategy="random")
        with pytest.raises(BackendError, match="No instances available"):
            lb.select_instance([])

    def test_select_random_returns_instance(self):
        """Test random selection returns one of the available instances."""
        lb = LoadBalancer(strategy="random")
        backend = MagicMock()
        instances = [
            InstanceInfo(host="host1", port=8000, backend=backend),
            InstanceInfo(host="host2", port=8000, backend=backend),
            InstanceInfo(host="host3", port=8000, backend=backend),
        ]
        selected = lb.select_instance(instances)
        assert selected in instances

    def test_select_random_single_instance(self):
        """Test random selection with a single instance."""
        lb = LoadBalancer(strategy="random")
        backend = MagicMock()
        instances = [InstanceInfo(host="host1", port=8000, backend=backend)]
        selected = lb.select_instance(instances)
        assert selected == instances[0]

    def test_round_robin_cycles_through_instances(self):
        """Test round-robin correctly cycles through all instances."""
        lb = LoadBalancer(strategy="round_robin")
        backend = MagicMock()
        instances = [
            InstanceInfo(host=f"host{i}", port=8000, backend=backend)
            for i in range(3)
        ]
        selections = [lb.select_instance(instances) for _ in range(6)]
        # Should cycle: host0, host1, host2, host0, host1, host2
        assert selections[0].host == "host0"
        assert selections[1].host == "host1"
        assert selections[2].host == "host2"
        assert selections[3].host == "host0"
        assert selections[4].host == "host1"
        assert selections[5].host == "host2"

    def test_round_robin_single_instance(self):
        """Test round-robin with single instance always returns same."""
        lb = LoadBalancer(strategy="round_robin")
        backend = MagicMock()
        instances = [InstanceInfo(host="only_host", port=8000, backend=backend)]
        for _ in range(5):
            assert lb.select_instance(instances).host == "only_host"

    def test_reset_clears_round_robin_counter(self):
        """Test reset resets the round-robin index."""
        lb = LoadBalancer(strategy="round_robin")
        backend = MagicMock()
        instances = [
            InstanceInfo(host=f"host{i}", port=8000, backend=backend)
            for i in range(3)
        ]
        # Advance the counter
        lb.select_instance(instances)
        lb.select_instance(instances)
        lb.reset()
        # After reset, should start from index 0
        assert lb.select_instance(instances).host == "host0"

    def test_default_strategy_is_random(self):
        """Test default strategy is random."""
        lb = LoadBalancer()
        assert lb._strategy == LoadBalancer.STRATEGY_RANDOM
