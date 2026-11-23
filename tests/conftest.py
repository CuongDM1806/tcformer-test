def pytest_configure(config):
    config.addinivalue_line("markers", "slow: trains small models on synthetic data")
