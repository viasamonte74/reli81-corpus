"""Final hidden states per request from an in-process vLLM engine.

Decode steps replay as CUDA graphs, so hooks inside the model never fire; the
return value of GPUModelRunner._model_forward survives replay, and its rows are
laid out in input_batch.req_ids order with each request's scheduled token
count. Measured on an H100 (2026-09-22): proofs built from these rows verify
against an HF prefill inside the prefill-against-prefill band.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
import os

import torch


def attribute_rows(
    order: Sequence[str], scheduled: Mapping[str, int], hidden: torch.Tensor
) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {}
    offset = 0
    for request_id in order:
        count = int(scheduled.get(request_id, 0))
        if count:
            rows[request_id] = hidden[offset : offset + count]
            offset += count
    if offset > hidden.shape[0]:
        raise ValueError(f"{offset} rows scheduled, {hidden.shape[0]} produced")
    return rows


def completion_rows(rows: torch.Tensor, prompt_len: int, total_len: int) -> torch.Tensor:
    """The rows that produced the completion. Async scheduling may run one extra
    step for a request its last token ended; anything else is refused."""
    expected = total_len - 1
    if rows.shape[0] not in (expected, expected + 1):
        raise ValueError(
            f"{rows.shape[0]} rows for {total_len} tokens: rows are missing "
            "(prefix caching must be off) or were attributed to the wrong request"
        )
    return rows[prompt_len - 1 : expected]


def _by_position(chunks: list[tuple[torch.Tensor, torch.Tensor | None]], limit: int | None) -> torch.Tensor:
    """A request's rows in position order. A position computed more than once
    (a rejected draft token, or a request recomputed after preemption) keeps its
    last computation: once a position's token is final, no later step computes
    it again."""
    if all(positions is None for _, positions in chunks):
        return torch.cat([rows for rows, _ in chunks], 0)
    if any(positions is None for _, positions in chunks):
        raise ValueError("rows recorded both with and without positions")
    for _, positions in chunks:
        if positions.numel() > 1 and not bool((positions[1:] - positions[:-1] == 1).all()):
            raise ValueError("a step's rows for one request are not consecutive positions")
    rows = torch.cat([rows for rows, _ in chunks], 0)
    positions = torch.cat([positions for _, positions in chunks], 0)
    end = int(positions.max()) + 1 if limit is None else limit
    keep = positions < end
    last = torch.full((end,), -1, dtype=torch.long)
    last.scatter_reduce_(0, positions[keep], torch.arange(len(positions))[keep], reduce="amax")
    missing = int((last < 0).sum())
    if missing:
        raise ValueError(f"no rows for {missing} of {end} positions")
    return rows[last]


class HiddenStateCapture:
    """Rows by request. ``non_blocking`` queues each step's copy to pinned host
    memory behind the forward instead of waiting for the forward to finish, so
    the scheduler keeps preparing the next step; the rows are the same bits.

    With ``positions`` given to ``record``, rows are reassembled by position,
    which speculative decoding (rows for rejected draft tokens) and preemption
    (a request computed again from its start) both need."""

    def __init__(self, non_blocking: bool = False) -> None:
        self._rows: dict[str, list[tuple[torch.Tensor, torch.Tensor | None]]] = {}
        self._non_blocking = non_blocking
        self._staged: list[tuple[torch.cuda.Event, torch.Tensor, torch.Tensor | None,
                                 list[tuple[str, int, int]]]] = []

    def record(self, order: Sequence[str], scheduled: Mapping[str, int], hidden: torch.Tensor,
               positions: torch.Tensor | None = None) -> None:
        if positions is not None and positions.dim() == 2:  # M-RoPE: equal rows for text
            positions = positions[0]
        spans, offset = [], 0
        for request_id in order:
            count = int(scheduled.get(request_id, 0))
            if count:
                spans.append((request_id, offset, count))
                offset += count
        if offset > hidden.shape[0]:
            raise ValueError(f"{offset} rows scheduled, {hidden.shape[0]} produced")
        if positions is not None and offset > positions.shape[0]:
            raise ValueError(f"{offset} rows scheduled, {positions.shape[0]} positions")
        if not (self._non_blocking and hidden.is_cuda):
            host = hidden[:offset].detach().to("cpu", torch.bfloat16)
            where = None if positions is None else positions[:offset].to("cpu", torch.long)
            self._keep(host, where, spans)
            return
        # Queued on the forward's stream, so it reads the output buffers before
        # the next step (or CUDA graph replay) can overwrite them.
        staged = torch.empty((offset, hidden.shape[1]), dtype=torch.bfloat16, pin_memory=True)
        staged.copy_(hidden[:offset].detach().to(torch.bfloat16), non_blocking=True)
        where = None
        if positions is not None:
            where = torch.empty((offset,), dtype=torch.long, pin_memory=True)
            where.copy_(positions[:offset].to(torch.long), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._staged.append((event, staged, where, spans))
        self._settle(wait=False)

    def _keep(self, host: torch.Tensor, where: torch.Tensor | None, spans) -> None:
        for request_id, start, count in spans:
            self._rows.setdefault(request_id, []).append((
                host[start : start + count].clone(),
                None if where is None else where[start : start + count].clone(),
            ))

    def _settle(self, *, wait: bool) -> None:
        while self._staged and (wait or self._staged[0][0].query()):
            event, staged, where, spans = self._staged.pop(0)
            event.synchronize()
            self._keep(staged, where, spans)

    def _match(self, request_id: str) -> str:
        self._settle(wait=True)
        # The runner suffixes engine ids ("0" becomes "0-ae415201").
        matches = [r for r in self._rows if r == request_id or r.startswith(request_id + "-")]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} captured requests match {request_id!r}")
        return matches[0]

    def for_request(self, request_id: str, limit: int | None = None) -> torch.Tensor:
        """``limit``: the positions wanted, when rows were recorded with them;
        later ones (an async step past the end, a final rejected draft) are
        dropped."""
        return _by_position(self._rows[self._match(request_id)], limit)

    def pop(self, request_id: str, limit: int | None = None) -> torch.Tensor:
        """Like ``for_request``, but forgets the rows: a long-lived miner process
        must not keep every completion's activations resident forever."""
        return _by_position(self._rows.pop(self._match(request_id)), limit)


def _check_engine_mode() -> None:
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("hidden-state capture needs VLLM_ENABLE_V1_MULTIPROCESSING=0")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("hidden-state capture is hooked on the V1 runner: set VLLM_USE_V2_MODEL_RUNNER=0")


def _record_step(capture: HiddenStateCapture, runner, scheduled: dict[str, int],
                 output, *, positions, positional: bool) -> None:
    """Take the runner's hidden states for this step into ``capture``."""
    if not scheduled:
        return
    hidden = output[0] if isinstance(output, tuple) else output
    batch = runner.input_batch
    # The positions the model was given, not the scheduler's: under async
    # scheduling a step is planned before the last one's draft tokens are
    # judged, and the runner corrects positions on the GPU.
    if not positional:
        positions = None
    capture.record(batch.req_ids[: batch.num_reqs], dict(scheduled), hidden, positions)
    scheduled.clear()


@contextlib.contextmanager
def capture_hidden_states(runner_cls=None, *, non_blocking: bool = False,
                          positional: bool = False) -> Iterator[HiddenStateCapture]:
    _check_engine_mode()
    if runner_cls is None:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as runner_cls
    capture = HiddenStateCapture(non_blocking=non_blocking)
    scheduled: dict[str, int] = {}
    original_execute = runner_cls.execute_model
    # Newer vLLM (post-0.10) routes the model call through ``_model_forward``;
    # 0.10 calls ``self.model(...)`` inside ``execute_model`` instead. Hook
    # whichever path this build has so proofs still get their activations.
    original_forward = getattr(runner_cls, "_model_forward", None)

    if original_forward is not None:
        def execute_model(self, scheduler_output, *args, **kwargs):
            scheduled.clear()
            scheduled.update(scheduler_output.num_scheduled_tokens)
            return original_execute(self, scheduler_output, *args, **kwargs)

        def _model_forward(self, *args, **kwargs):
            output = original_forward(self, *args, **kwargs)
            positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
            _record_step(capture, self, scheduled, output, positions=positions,
                         positional=positional)
            return output

        runner_cls.execute_model = execute_model
        runner_cls._model_forward = _model_forward
        try:
            yield capture
        finally:
            runner_cls.execute_model = original_execute
            runner_cls._model_forward = original_forward
        return

    def execute_model(self, scheduler_output, *args, **kwargs):
        scheduled.clear()
        scheduled.update(scheduler_output.num_scheduled_tokens)
        model = self.model
        original_model_forward = model.forward

        def model_forward(*margs, **mkwargs):
            output = original_model_forward(*margs, **mkwargs)
            positions = mkwargs.get("positions", margs[1] if len(margs) > 1 else None)
            _record_step(capture, self, scheduled, output, positions=positions,
                         positional=positional)
            return output

        model.forward = model_forward
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            model.forward = original_model_forward

    runner_cls.execute_model = execute_model
    try:
        yield capture
    finally:
        runner_cls.execute_model = original_execute
