import importlib
import sys
import types

import torch


def _install_vllm_stubs():
    if "vllm" in sys.modules:
        return

    def ensure_module(name: str):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    vllm = ensure_module("vllm")
    vllm.distributed = ensure_module("vllm.distributed")
    vllm.distributed.parallel_state = ensure_module("vllm.distributed.parallel_state")
    vllm.config = ensure_module("vllm.config")
    vllm.sequence = ensure_module("vllm.sequence")
    vllm.sampling_params = ensure_module("vllm.sampling_params")
    vllm.lora = ensure_module("vllm.lora")
    vllm.lora.request = ensure_module("vllm.lora.request")
    vllm.v1 = ensure_module("vllm.v1")
    vllm.v1.attention = ensure_module("vllm.v1.attention")
    vllm.v1.attention.backends = ensure_module("vllm.v1.attention.backends")
    vllm.v1.attention.backends.registry = ensure_module("vllm.v1.attention.backends.registry")
    vllm.v1.core = ensure_module("vllm.v1.core")
    vllm.v1.core.kv_cache_manager = ensure_module("vllm.v1.core.kv_cache_manager")
    vllm.v1.core.kv_cache_utils = ensure_module("vllm.v1.core.kv_cache_utils")
    vllm.v1.core.sched = ensure_module("vllm.v1.core.sched")
    vllm.v1.core.sched.output = ensure_module("vllm.v1.core.sched.output")
    vllm.v1.kv_cache_interface = ensure_module("vllm.v1.kv_cache_interface")
    vllm.v1.request = ensure_module("vllm.v1.request")
    vllm.v1.worker = ensure_module("vllm.v1.worker")
    vllm.v1.worker.gpu_model_runner = ensure_module("vllm.v1.worker.gpu_model_runner")
    vllm.v1.worker.workspace = ensure_module("vllm.v1.worker.workspace")

    class Dummy:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DummyIntermediateTensors(dict):
        pass

    class DummyGPUModelRunner:
        def __init__(self, *args, **kwargs):
            pass

        def execute_model(self, *args, **kwargs):
            return None

        def sample_tokens(self, *args, **kwargs):
            return None

    class DummyEnum:
        TORCH_SDPA = "torch"
        FLASHINFER = "flashinfer"
        TRITON_ATTN = "triton"
        FLASH_ATTN = "flash-attn"

    vllm.sequence.IntermediateTensors = DummyIntermediateTensors
    vllm.sampling_params.SamplingParams = Dummy
    vllm.sampling_params.StructuredOutputsParams = Dummy
    vllm.distributed.parallel_state.GroupCoordinator = Dummy
    vllm.config.AttentionConfig = Dummy
    vllm.config.CacheConfig = Dummy
    vllm.config.DeviceConfig = Dummy
    vllm.config.LoadConfig = Dummy
    vllm.config.LoRAConfig = Dummy
    vllm.config.ModelConfig = Dummy
    vllm.config.ParallelConfig = Dummy
    vllm.config.SchedulerConfig = Dummy
    vllm.config.VllmConfig = Dummy
    vllm.config.set_current_vllm_config = lambda *args, **kwargs: None
    vllm.v1.attention.backends.registry.AttentionBackendEnum = DummyEnum
    vllm.lora.request.LoRARequest = Dummy
    vllm.v1.core.kv_cache_manager.KVCacheManager = Dummy
    vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config = (
        lambda *args, **kwargs: Dummy()
    )
    vllm.v1.core.kv_cache_utils.get_kv_cache_configs = lambda *args, **kwargs: [Dummy()]
    vllm.v1.kv_cache_interface.FullAttentionSpec = Dummy
    vllm.v1.kv_cache_interface.KVCacheConfig = Dummy
    vllm.v1.kv_cache_interface.KVCacheGroupSpec = Dummy
    vllm.v1.kv_cache_interface.KVCacheTensor = Dummy
    vllm.v1.core.sched.output.CachedRequestData = Dummy
    vllm.v1.core.sched.output.NewRequestData = Dummy
    vllm.v1.core.sched.output.SchedulerOutput = Dummy
    vllm.v1.request.Request = Dummy
    vllm.v1.worker.gpu_model_runner.GPUModelRunner = DummyGPUModelRunner
    vllm.v1.worker.workspace.current_workspace_manager = lambda: Dummy(
        _ensure_workspace_size=lambda size: None
    )
    vllm.v1.worker.workspace.init_workspace_manager = lambda device: None


_install_vllm_stubs()

model_runner_module = importlib.import_module("parallax.vllm.model_runner")
vllm_executor_module = importlib.import_module("parallax.server.executor.vllm_executor")

ParallaxVLLMModelRunner = model_runner_module.ParallaxVLLMModelRunner
VLLMExecutor = vllm_executor_module.VLLMExecutor


def test_vllm_model_runner_non_decoding_returns_stable_tuple(monkeypatch):
    runner = ParallaxVLLMModelRunner.__new__(ParallaxVLLMModelRunner)
    runner.is_first_peer = True
    runner.intermediate_tensors = None
    runner.execute_model_state = None
    returned_state = {"hidden_states": "hs", "residual": "r"}

    execute_calls = []
    sample_calls = []

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: execute_calls.append(
            (scheduler_output, intermediate_tensors)
        )
        or returned_state,
    )
    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "sample_tokens",
        lambda self, grammar_output=None: sample_calls.append(grammar_output),
    )

    result = runner.execute_model(
        scheduler_output="scheduler-output",
        intermediate_tensors="proxy",
        return_decoded_tokens=False,
    )

    assert result == (returned_state, None, None, None, None)
    assert runner.execute_model_state is None
    assert execute_calls == [("scheduler-output", "proxy")]
    assert sample_calls == []


def test_vllm_model_runner_non_decoding_supports_legacy_in_place_state(monkeypatch):
    runner = ParallaxVLLMModelRunner.__new__(ParallaxVLLMModelRunner)
    runner.is_first_peer = True
    runner.intermediate_tensors = None
    runner.execute_model_state = types.SimpleNamespace(hidden_states="legacy-hs")

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: None,
    )
    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "sample_tokens",
        lambda self, grammar_output=None: None,
    )

    result = runner.execute_model(
        scheduler_output="scheduler-output",
        intermediate_tensors="proxy",
        return_decoded_tokens=False,
    )

    assert result == (types.SimpleNamespace(hidden_states="legacy-hs"), None, None, None, None)
    assert runner.execute_model_state is None


def test_vllm_model_runner_raises_when_execution_state_is_missing(monkeypatch):
    runner = ParallaxVLLMModelRunner.__new__(ParallaxVLLMModelRunner)
    runner.is_first_peer = True
    runner.intermediate_tensors = None
    runner.execute_model_state = None

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: None,
    )

    try:
        runner.execute_model(
            scheduler_output="scheduler-output",
            intermediate_tensors="proxy",
            return_decoded_tokens=False,
        )
    except RuntimeError as exc:
        assert "returned no intermediate tensors" in str(exc)
    else:
        raise AssertionError("Expected execute_model to fail when vLLM returns no state")


def test_vllm_model_runner_decoding_samples_when_execute_model_returns_none(monkeypatch):
    runner = ParallaxVLLMModelRunner.__new__(ParallaxVLLMModelRunner)
    runner.is_first_peer = True
    runner.intermediate_tensors = None
    runner.execute_model_state = types.SimpleNamespace(logits="legacy-logits")

    sample_output = types.SimpleNamespace(
        _sampled_token_ids=torch.tensor([[7], [9]], dtype=torch.long),
        sampled_token_ids_cpu=torch.tensor([7, 9], dtype=torch.long),
    )
    sample_calls = []

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: None,
    )
    def _sample_tokens(self, grammar_output=None):
        sample_calls.append(grammar_output)
        self.execute_model_state = None
        return sample_output

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "sample_tokens",
        _sample_tokens,
    )

    result = runner.execute_model(
        scheduler_output="scheduler-output",
        intermediate_tensors="proxy",
        return_decoded_tokens=True,
    )

    assert result[0] is sample_output
    assert torch.equal(result[1], torch.tensor([[7], [9]], dtype=torch.long))
    assert torch.equal(result[2], torch.tensor([7, 9], dtype=torch.long))
    assert result[3] is sample_output
    assert result[4] == "legacy-logits"
    assert sample_calls == [None]
    assert runner.execute_model_state is None


def test_vllm_model_runner_decoding_uses_direct_output_without_sampling(monkeypatch):
    runner = ParallaxVLLMModelRunner.__new__(ParallaxVLLMModelRunner)
    runner.is_first_peer = True
    runner.intermediate_tensors = None
    runner.execute_model_state = None
    sample_calls = []
    direct_output = types.SimpleNamespace(sampled_token_ids=[[3], [4]], logits="direct-logits")

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: direct_output,
    )
    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "sample_tokens",
        lambda self, grammar_output=None: sample_calls.append(grammar_output),
    )

    result = runner.execute_model(
        scheduler_output="scheduler-output",
        intermediate_tensors="proxy",
        return_decoded_tokens=True,
    )

    assert result[0] is direct_output
    assert result[1] == [[3], [4]]
    assert torch.equal(result[2], torch.tensor([3, 4], dtype=torch.long))
    assert result[3] is direct_output
    assert result[4] == "direct-logits"
    assert sample_calls == []
    assert runner.execute_model_state is None


def test_vllm_model_runner_decoding_resolves_async_output_once(monkeypatch):
    runner = ParallaxVLLMModelRunner.__new__(ParallaxVLLMModelRunner)
    runner.is_first_peer = True
    runner.intermediate_tensors = None
    runner.execute_model_state = None

    direct_output = types.SimpleNamespace(sampled_token_ids=[[11]], logits="async-logits")

    class AsyncOutput:
        def __init__(self):
            self.calls = 0

        def get_output(self):
            self.calls += 1
            return direct_output

    async_output = AsyncOutput()
    sample_calls = []

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: async_output,
    )
    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "sample_tokens",
        lambda self, grammar_output=None: sample_calls.append(grammar_output),
    )

    result = runner.execute_model(
        scheduler_output="scheduler-output",
        intermediate_tensors="proxy",
        return_decoded_tokens=True,
    )

    assert result[0] is direct_output
    assert result[1] == [[11]]
    assert torch.equal(result[2], torch.tensor([11], dtype=torch.long))
    assert result[3] is async_output
    assert result[4] == "async-logits"
    assert async_output.calls == 1
    assert sample_calls == []
    assert runner.execute_model_state is None


def test_vllm_model_runner_non_decoding_clears_pending_state_when_execute_returns_none(monkeypatch):
    runner = ParallaxVLLMModelRunner.__new__(ParallaxVLLMModelRunner)
    runner.is_first_peer = False
    runner.intermediate_tensors = "existing-buffer"
    runner.execute_model_state = types.SimpleNamespace(hidden_states="broadcast-hidden")

    monkeypatch.setattr(
        model_runner_module.GPUModelRunner,
        "execute_model",
        lambda self, scheduler_output, intermediate_tensors: None,
    )

    result = runner.execute_model(
        scheduler_output="scheduler-output",
        intermediate_tensors="proxy",
        return_decoded_tokens=False,
    )

    assert result == (types.SimpleNamespace(hidden_states="broadcast-hidden"), None, None, None, None)
    assert runner.execute_model_state is None


def test_vllm_executor_non_last_peer_returns_hidden_states():
    executor = VLLMExecutor.__new__(VLLMExecutor)
    executor.model_runner = types.SimpleNamespace(
        execute_model=lambda **kwargs: (
            types.SimpleNamespace(hidden_states="forwarded-hidden-states"),
            None,
            None,
            None,
            None,
        )
    )

    output = executor.process_batch(
        {
            "scheduler_output": "scheduler-output",
            "pp_proxy_tensors": "proxy",
            "requests": [],
        },
        return_decoded_tokens=False,
    )

    assert output == {"hidden_states": "forwarded-hidden-states", "probs": None}


def test_vllm_executor_non_last_peer_accepts_intermediate_tensors_mapping():
    executor = VLLMExecutor.__new__(VLLMExecutor)
    executor.model_runner = types.SimpleNamespace(
        execute_model=lambda **kwargs: (
            {"hidden_states": "forwarded-from-mapping", "residual": "ignored"},
            None,
            None,
            None,
            None,
        )
    )

    output = executor.process_batch(
        {
            "scheduler_output": "scheduler-output",
            "pp_proxy_tensors": "proxy",
            "requests": [],
        },
        return_decoded_tokens=False,
    )

    assert output == {"hidden_states": "forwarded-from-mapping", "probs": None}


def test_vllm_executor_non_last_peer_raises_when_hidden_states_missing():
    executor = VLLMExecutor.__new__(VLLMExecutor)
    executor.model_runner = types.SimpleNamespace(
        execute_model=lambda **kwargs: (
            {"residual": "missing-hidden"},
            None,
            None,
            None,
            None,
        )
    )

    try:
        executor.process_batch(
            {
                "scheduler_output": "scheduler-output",
                "pp_proxy_tensors": "proxy",
                "requests": [],
            },
            return_decoded_tokens=False,
        )
    except RuntimeError as exc:
        assert "does not expose hidden_states" in str(exc)
    else:
        raise AssertionError("Expected process_batch to fail when hidden_states are missing")


def test_vllm_executor_last_peer_returns_sampled_tokens():
    executor = VLLMExecutor.__new__(VLLMExecutor)
    executor.model_runner = types.SimpleNamespace(
        execute_model=lambda **kwargs: (
            types.SimpleNamespace(hidden_states="unused"),
            torch.tensor([[7], [9]], dtype=torch.long),
            torch.tensor([7, 9], dtype=torch.long),
            types.SimpleNamespace(),
            None,
        ),
        input_batch=None,
    )

    output = executor.process_batch(
        {
            "scheduler_output": "scheduler-output",
            "pp_proxy_tensors": None,
            "requests": [],
        },
        return_decoded_tokens=True,
    )

    assert torch.equal(output["hidden_states"], torch.tensor([7, 9], dtype=torch.long))
    assert output["probs"] is None
