from parallax.server.server_info import (
    AppleSiliconHardwareInfo,
    HardwareInfo,
    detect_node_hardware,
)


def test_detect_node_hardware_applies_test_overrides(monkeypatch):
    monkeypatch.setattr(
        HardwareInfo,
        "detect",
        staticmethod(
            lambda: AppleSiliconHardwareInfo(
                num_gpus=1,
                total_ram_gb=24.0,
                chip="Apple M4 Pro",
                tflops_fp16=17.04,
            )
        ),
    )
    monkeypatch.setenv("PARALLAX_TEST_OVERRIDE_MEMORY_GB", "1.0")
    monkeypatch.setenv("PARALLAX_TEST_OVERRIDE_MEMORY_BANDWIDTH_GBPS", "42.5")
    monkeypatch.setenv("PARALLAX_TEST_OVERRIDE_GPU_NAME", "Membership Smoke Worker")

    payload = detect_node_hardware("node-1")

    assert payload["node_id"] == "node-1"
    assert payload["device"] == "mlx"
    assert payload["memory_gb"] == 1.0
    assert payload["memory_bandwidth_gbps"] == 42.5
    assert payload["gpu_name"] == "Membership Smoke Worker"
