from __future__ import annotations

import pytest

from arouter.bench import SimulatedFactory
from arouter.cache import LocalCacheBackend, PrefixCache
from arouter.cascade import CascadeRunner, RunConfig
from arouter.models.verifier import SimulatedVerifier
from arouter.pricing import LADDER
from arouter.suite import Suite, load_suite


@pytest.fixture(scope="session")
def suite() -> Suite:
    return load_suite()


@pytest.fixture
def runner(suite: Suite) -> CascadeRunner:
    return CascadeRunner(
        suite,
        SimulatedFactory(seed=0),
        prefix_cache=PrefixCache(backend=LocalCacheBackend()),
        verifier_factory=lambda tier, run_id: SimulatedVerifier(tier=tier, seed=0),
    )


@pytest.fixture
def cascade_config() -> RunConfig:
    return RunConfig(name="test_cascade", tiers=LADDER[:1] + LADDER[-1:], policy="schema_validation")
