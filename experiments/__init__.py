"""Long-running measurements that are not part of the test suite.

Nothing here asserts and nothing here runs under ``pytest`` (``pytest.ini``
collects ``tests/`` only).  Each module measures something over many random
draws, writes one CSV row per run, and leaves the reading of those rows to
``experiments.summarize``.
"""
