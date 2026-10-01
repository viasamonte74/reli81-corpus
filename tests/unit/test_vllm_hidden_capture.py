"""Row attribution for the vLLM capture, tested on a fake runner: the real one
needs a GPU, and the logic that can go wrong is the bookkeeping."""

from types import SimpleNamespace

import pytest
import torch

from reliquary.miner.vllm_hidden_capture import (
    HiddenStateCapture,
    attribute_rows,
    capture_hidden_states,
    completion_rows,
)


def test_rows_follow_the_batch_order_and_scheduled_counts():
    hidden = torch.arange(6).float().unsqueeze(1)
    rows = attribute_rows(["b", "a", "c"], {"a": 1, "b": 3, "c": 0}, hidden)
    assert rows["b"].flatten().tolist() == [0, 1, 2]
    assert rows["a"].flatten().tolist() == [3]
    assert "c" not in rows


def test_more_scheduled_rows_than_produced_is_refused():
    with pytest.raises(ValueError):
        attribute_rows(["a"], {"a": 5}, torch.zeros(3, 1))


def test_completion_rows_start_at_the_last_prompt_position():
    rows = torch.arange(9).float().unsqueeze(1)          # total_len 10 -> 9 rows
    assert completion_rows(rows, 4, 10).flatten().tolist() == [3, 4, 5, 6, 7, 8]


def test_one_surplus_row_from_async_scheduling_is_dropped():
    rows = torch.arange(10).float().unsqueeze(1)
    assert completion_rows(rows, 4, 10).flatten().tolist() == [3, 4, 5, 6, 7, 8]


def test_missing_rows_are_refused():
    with pytest.raises(ValueError, match="prefix caching"):
        completion_rows(torch.zeros(5, 1), 4, 10)


class _FakeRunner:
    def __init__(self, steps):
        self._steps = list(steps)
        self.input_batch = SimpleNamespace(req_ids=[], num_reqs=0)

    def execute_model(self, scheduler_output):
        return self._model_forward()

    def _model_forward(self):
        order, hidden = self._steps.pop(0)
        self.input_batch.req_ids = order
        self.input_batch.num_reqs = len(order)
        return hidden


def _env(monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")


def test_the_patch_records_every_step_and_is_removed_after(monkeypatch):
    _env(monkeypatch)
    runner = _FakeRunner([
        (["0-x", "1-y"], torch.tensor([[1.0], [2.0], [3.0]])),
        (["0-x", "1-y"], torch.tensor([[4.0], [5.0]])),
    ])
    original = _FakeRunner.execute_model
    with capture_hidden_states(_FakeRunner) as capture:
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 2, "1-y": 1}))
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 1, "1-y": 1}))
    assert capture.for_request("0").flatten().tolist() == [1.0, 2.0, 4.0]
    assert capture.for_request("1").flatten().tolist() == [3.0, 5.0]
    assert _FakeRunner.execute_model is original


@pytest.mark.parametrize(
    "variable,value",
    [("VLLM_ENABLE_V1_MULTIPROCESSING", "1"), ("VLLM_USE_V2_MODEL_RUNNER", "1")],
)
def test_an_unsupported_engine_mode_is_refused(monkeypatch, variable, value):
    _env(monkeypatch)
    monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError):
        with capture_hidden_states(_FakeRunner):
            pass


def test_pop_returns_and_forgets_the_request(monkeypatch):
    _env(monkeypatch)
    runner = _FakeRunner([(["0-x", "1-y"], torch.tensor([[1.0], [2.0], [3.0]]))])
    with capture_hidden_states(_FakeRunner) as capture:
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 2, "1-y": 1}))
        popped = capture.pop("0")
        assert popped.flatten().tolist() == [1.0, 2.0]
        with pytest.raises(KeyError):
            capture.pop("0")
        # The other request is untouched by popping the first.
        assert capture.for_request("1").flatten().tolist() == [3.0]


def test_non_blocking_on_cpu_tensors_records_like_blocking():
    capture = HiddenStateCapture(non_blocking=True)
    capture.record(["0-x", "1-y"], {"0-x": 2, "1-y": 1}, torch.tensor([[1.0], [2.0], [3.0]]))
    assert capture.pop("0").flatten().tolist() == [1.0, 2.0]
    assert capture.pop("1").flatten().tolist() == [3.0]


def _step(capture, positions, values):
    capture.record(["0-x"], {"0-x": len(positions)}, torch.tensor(values).float().unsqueeze(1),
                   torch.tensor(positions))


def test_rejected_draft_rows_give_way_to_the_step_that_recomputes_them():
    capture = HiddenStateCapture()
    _step(capture, [0, 1, 2, 3], [10, 11, 12, 13])     # prefill
    _step(capture, [4, 5, 6], [14, 15, -1])            # draft at 6 rejected
    _step(capture, [6, 7, 8], [16, 17, -1])            # 8: a final rejected draft
    assert capture.pop("0", limit=8).flatten().tolist() == [10, 11, 12, 13, 14, 15, 16, 17]


def test_a_request_recomputed_after_preemption_keeps_the_recomputed_rows():
    capture = HiddenStateCapture()
    _step(capture, [0, 1, 2], [-1, -1, -1])
    _step(capture, [3], [-1])
    _step(capture, [0, 1, 2, 3, 4], [10, 11, 12, 13, 14])
    assert capture.pop("0").flatten().tolist() == [10, 11, 12, 13, 14]


def test_a_position_never_computed_is_refused():
    capture = HiddenStateCapture()
    _step(capture, [0, 1], [10, 11])
    _step(capture, [3, 4], [13, 14])
    with pytest.raises(ValueError, match="no rows"):
        capture.pop("0", limit=5)


def test_a_step_whose_positions_are_not_consecutive_is_refused():
    capture = HiddenStateCapture()
    _step(capture, [0, 2], [10, 12])
    with pytest.raises(ValueError, match="consecutive"):
        capture.pop("0")


def test_the_patch_passes_the_positions_the_model_was_given(monkeypatch):
    _env(monkeypatch)

    class _Runner(_FakeRunner):
        def execute_model(self, scheduler_output, positions):
            return self._model_forward(positions=positions)

        def _model_forward(self, positions=None):
            return super()._model_forward()

    runner = _Runner([
        (["0-x"], torch.tensor([[1.0], [2.0], [3.0]])),
        (["0-x"], torch.tensor([[9.0], [4.0]])),
    ])
    with capture_hidden_states(_Runner, positional=True) as capture:
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 3}), torch.tensor([0, 1, 2]))
        runner.execute_model(SimpleNamespace(num_scheduled_tokens={"0-x": 2}), torch.tensor([2, 3]))
    assert capture.pop("0").flatten().tolist() == [1.0, 2.0, 9.0, 4.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_non_blocking_rows_are_the_blocking_bits_even_when_the_buffer_is_reused():
    # CUDA graph replay writes every step's output into the same buffer.
    buffer = torch.empty((6, 64), device="cuda", dtype=torch.bfloat16)
    where = torch.empty((6,), device="cuda", dtype=torch.long)
    blocking, fast = HiddenStateCapture(), HiddenStateCapture(non_blocking=True)
    steps = [(["0-x", "1-y"], {"0-x": 4, "1-y": 1}, [0, 1, 2, 3, 0]),
             (["1-y", "0-x"], {"0-x": 1, "1-y": 1}, [1, 4]),
             (["1-y", "0-x"], {"0-x": 1, "1-y": 1}, [2, 5])]
    for order, scheduled, positions in steps:
        buffer.normal_()
        where[: len(positions)] = torch.tensor(positions)
        blocking.record(order, scheduled, buffer, where)
        fast.record(order, scheduled, buffer, where)
    buffer.fill_(float("nan"))
    where.fill_(-1)
    for request in ("0", "1"):
        assert torch.equal(fast.pop(request), blocking.pop(request))
