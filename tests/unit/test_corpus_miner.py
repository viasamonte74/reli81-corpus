"""The miner walks its own order, proves what it generated, and resyncs on refusal."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from reliquary.corpus.encoding import prompt_token_ids
from reliquary.corpus.walk import walk_index
from reliquary.miner.corpus_miner import (
    Backlog,
    CorpusMinerHalted,
    CorpusPermanentFailure,
    CorpusTransientFailure,
    Generation,
    VllmGenerator,
    build_submission,
    mine_steps,
    mine_window,
)

EOS = 99


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)


def _job(prompt_count=50, n=2, prompt_start=0):
    return SimpleNamespace(job_id="math-v1", prompt_count=prompt_count,
                           prompt_start=prompt_start, eos_token_id=EOS,
                           checkpoint_sha256="a" * 64, sampling=SimpleNamespace(n=n),
                           prompt_order="miner_walk")


class _Generator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt_ids, n):
        self.prompts.append(prompt_ids)
        return [Generation(tokens=[104, 105, EOS], proofs=["AAAA"]) for _ in range(n)]


class _Client:
    def __init__(self, answers):
        self.answers = list(answers)
        self.submitted = []
        self.cursor_reads = 0
        self.position = 0

    def cursor(self, hotkey):
        self.cursor_reads += 1
        return self.position

    def submit(self, body):
        self.submitted.append(body)
        answer = self.answers.pop(0)
        if answer == "accepted":
            self.position += 1
        return {"reason": answer, "accepted": answer == "accepted"}


def test_the_submission_carries_the_walk_prompt_and_its_text():
    body = build_submission(job=_job(), hotkey="5Hot", cursor=0, prompt_index=7, rendered_prompt="q7",
                            generations=[Generation([104, 105, EOS], ["AAAA"])],
                            tokenizer=_Tokenizer(), sign=lambda b: "sig")
    assert body["prompt_index"] == 7 and body["signature"] == "sig"
    assert body["completions"] == [{"tokens": [104, 105, EOS], "text": "hi", "proofs": ["AAAA"]}]


def test_the_miner_follows_its_own_walk():
    client, generator = _Client(["accepted"] * 3), _Generator()
    mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator, tokenizer=_Tokenizer(),
               render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=3)
    assert [b["prompt_index"] for b in client.submitted] == [walk_index("math-v1", "5Hot", c, 50) for c in range(3)]
    assert [b["cursor"] for b in client.submitted] == [0, 1, 2]


def test_a_started_job_renders_and_submits_the_source_row():
    """The miner renders and submits the SOURCE index, the one the route's
    fidelity check renders: the walk shifted by the job's start."""
    rendered = []
    client, generator = _Client(["accepted"] * 4), _Generator()

    def render(index):
        rendered.append(index)
        return f"q{index}"

    mine_steps(job=_job(prompt_start=7000), hotkey="5Hot", client=client, generator=generator,
               tokenizer=_Tokenizer(), render=render, sign=lambda b: "sig", max_steps=4)
    expected = [7000 + walk_index("math-v1", "5Hot", c, 50) for c in range(4)]
    assert rendered == expected
    assert [b["prompt_index"] for b in client.submitted] == expected
    assert [b["rendered_prompt"] for b in client.submitted] == [f"q{i}" for i in expected]
    assert all(7000 <= i < 7050 for i in expected)


def test_a_refused_step_resynchronises_the_cursor():
    client = _Client(["prompt_full", "accepted"])
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=2)
    assert counts == {"prompt_full": 1, "accepted": 1}
    assert client.cursor_reads >= 2


def test_a_complete_job_stops_the_miner():
    client = _Client(["job_complete", "accepted"])
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=5)
    assert counts == {"job_complete": 1} and len(client.submitted) == 1


@pytest.mark.parametrize("reason", ["hotkey_not_registered", "miner_banned"])
def test_a_refusal_no_retry_can_change_stops_the_miner(reason):
    """Generating on would burn the card for nothing: stop and say why."""
    client = _Client([reason, "accepted"])
    with pytest.raises(CorpusMinerHalted, match=reason):
        mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                   tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                   max_steps=5)
    assert len(client.submitted) == 1


def test_miner_and_auditor_tokenize_the_prompt_identically():
    generator = _Generator()
    mine_steps(job=_job(), hotkey="5Hot", client=_Client(["accepted"]), generator=generator,
               tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig", max_steps=1)
    index = walk_index("math-v1", "5Hot", 0, 50)
    assert generator.prompts == [prompt_token_ids(_Tokenizer(), f"q{index}")]


# --- fix round 1: transient/permanent HTTP failures, and a generation failure ---


class _FlakyOnceClient:
    """``submit`` fails transiently once, then accepts."""

    def __init__(self):
        self.position = 0
        self.cursor_reads = 0
        self.submit_calls: list[dict] = []

    def cursor(self, hotkey):
        self.cursor_reads += 1
        return self.position

    def submit(self, body):
        self.submit_calls.append(body)
        if len(self.submit_calls) == 1:
            raise CorpusTransientFailure("503 ledger contention")
        self.position += 1
        return {"reason": "accepted", "accepted": True}


def test_a_transient_submit_failure_retries_the_identical_body():
    client, generator, sleeps = _FlakyOnceClient(), _Generator(), []
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                        max_steps=1, sleep=sleeps.append)
    assert len(generator.prompts) == 1, "one generation, not one per retry"
    assert len(client.submit_calls) == 2
    assert client.submit_calls[0] == client.submit_calls[1], "the SAME signed body, not a fresh one"
    assert counts == {"accepted": 1}
    assert sleeps == [1.0]


class _CursorFlakyOnceClient:
    """The initial ``cursor`` read fails transiently once, then succeeds."""

    def __init__(self):
        self.position = 0
        self.cursor_calls = 0
        self.submit_calls: list[dict] = []

    def cursor(self, hotkey):
        self.cursor_calls += 1
        if self.cursor_calls == 1:
            raise CorpusTransientFailure("timeout reading cursor")
        return self.position

    def submit(self, body):
        self.submit_calls.append(body)
        self.position += 1
        return {"reason": "accepted", "accepted": True}


def test_a_transient_cursor_failure_retries_and_does_not_crash():
    client, generator, sleeps = _CursorFlakyOnceClient(), _Generator(), []
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                        max_steps=1, sleep=sleeps.append)
    assert client.cursor_calls == 2
    assert counts == {"accepted": 1}
    assert sleeps == [1.0]


class _AlwaysFailingClient:
    """``submit`` always answers with a permanent failure (e.g. a 404 the
    job route sends once the job is gone)."""

    def __init__(self):
        self.cursor_reads = 0
        self.submit_calls: list[dict] = []

    def cursor(self, hotkey):
        self.cursor_reads += 1
        return 0

    def submit(self, body):
        self.submit_calls.append(body)
        raise CorpusPermanentFailure(
            "404 corpus_job_unknown", status=404, detail={"detail": "corpus_job_unknown"}
        )


def test_a_permanent_failure_stops_the_miner_after_k_consecutive():
    client, generator, sleeps = _AlwaysFailingClient(), _Generator(), []
    with pytest.raises(CorpusMinerHalted) as excinfo:
        mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                   tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                   max_steps=None, sleep=sleeps.append, max_consecutive_failures=3)
    assert len(client.submit_calls) == 3
    assert len(generator.prompts) == 1, "the retries resend the SAME body, no fresh generation per attempt"
    assert excinfo.value.counts == {"permanent_failure": 3}
    assert len(sleeps) == 2, "no backoff after the failure that crosses the threshold"


class _FlakyGenerator:
    """Raises once (as ``completion_rows`` would on a vLLM preemption), then
    generates normally."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt_ids, n):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("3 rows for 5 tokens: rows are missing")
        return [Generation(tokens=[104, 105, EOS], proofs=["AAAA"]) for _ in range(n)]


def test_a_generation_failure_drops_the_step_and_continues():
    client, generator = _Client(["accepted"]), _FlakyGenerator()
    counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=generator,
                        tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                        max_steps=2, sleep=lambda s: None)
    assert generator.calls == 2
    assert len(client.submitted) == 1, "the failed step submits nothing"
    assert counts == {"generation_failed": 1, "accepted": 1}
    assert client.cursor_reads == 2, "one initial read, one resync after the dropped step"


class _WindowGenerator:
    """Finishes the newest and the oldest request in turn, so later cursors
    complete before earlier ones, and remembers the most requests in flight."""

    def __init__(self, fail_once=()):
        self.live = []
        self.issued = 0
        self.steps = 0
        self.peak = 0
        self.fail_once = set(fail_once)
        self.cancelled = []

    def start(self, prompt_ids, n):
        request_ids = []
        for _ in range(n):
            self.issued += 1
            request_ids.append(f"{self.issued}-x")
        self.live.extend(request_ids)
        self.peak = max(self.peak, len(self.live))
        return request_ids

    def busy(self):
        return bool(self.live)

    def step(self):
        self.steps += 1
        return [(self.live.pop(-1 if self.steps % 2 else 0), [104, 105, EOS])]

    def finish(self, request_id, prompt_len, tokens):
        if request_id in self.fail_once:
            self.fail_once.discard(request_id)
            raise ValueError("3 rows for 5 tokens: rows are missing")
        return Generation(tokens, ["AAAA"])

    def cancel(self, request_ids):
        self.cancelled.extend(request_ids)
        self.live = [r for r in self.live if r not in request_ids]


def _mine_window(client, generator, window, max_steps, n=2):
    return mine_window(job=_job(n=n), hotkey="5Hot", client=client, generator=generator,
                       tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                       window=window, max_steps=max_steps, sleep=lambda s: None)


def test_the_window_submits_in_cursor_order_and_never_exceeds_its_size():
    client, generator = _Client(["accepted"] * 6), _WindowGenerator()
    counts = _mine_window(client, generator, window=3, max_steps=6)
    assert [b["cursor"] for b in client.submitted] == [0, 1, 2, 3, 4, 5]
    assert [b["prompt_index"] for b in client.submitted] == [
        walk_index("math-v1", "5Hot", c, 50) for c in range(6)]
    assert all(len(b["completions"]) == 2 for b in client.submitted)
    assert counts == {"accepted": 6}
    assert generator.peak == 3 * 2, "the window bounds requests in flight, n per prompt"
    assert not generator.live


def test_a_prompt_refused_without_moving_the_ledger_is_generated_again():
    """The regeneration is not a new step: every lookahead prompt still lands."""
    client = _Client(["bad_termination"] + ["accepted"] * 3)
    counts = _mine_window(client, _WindowGenerator(), window=2, max_steps=3, n=1)
    assert [b["cursor"] for b in client.submitted] == [0, 0, 1, 2]
    assert counts == {"bad_termination": 1, "accepted": 3}


def test_a_failed_generation_is_regenerated_at_its_cursor():
    client, generator = _Client(["accepted"] * 3), _WindowGenerator(fail_once={"1-x"})
    counts = _mine_window(client, generator, window=2, max_steps=3, n=1)
    assert [b["cursor"] for b in client.submitted] == [0, 1, 2]
    assert counts == {"generation_failed": 1, "accepted": 3}
    assert generator.peak <= 2


def test_a_complete_job_stops_the_window_and_cancels_what_is_generating():
    client, generator = _Client(["job_complete"]), _WindowGenerator()
    counts = _mine_window(client, generator, window=3, max_steps=10, n=1)
    assert counts == {"job_complete": 1} and len(client.submitted) == 1
    assert not generator.live and generator.cancelled


class _RoomGenerator(_WindowGenerator):
    """KV room for ``budget`` requests, whatever the window allows."""

    def __init__(self, budget):
        super().__init__()
        self.budget = budget

    def room(self, prompt_len, n):
        return len(self.live) + n <= self.budget


@pytest.mark.parametrize("budget, n", [(3, 1), (4, 2), (0, 1)])
def test_room_admits_prompts_beside_the_window(budget, n):
    """Room caps what generates below the window; with nothing generating a
    prompt is admitted anyway, since any one request fits the cache."""
    client, generator = _Client(["accepted"] * 6), _RoomGenerator(budget)
    counts = _mine_window(client, generator, window=10, max_steps=6, n=n)
    assert [b["cursor"] for b in client.submitted] == list(range(6))
    assert counts == {"accepted": 6}
    assert generator.peak == max(budget, n)
    assert not generator.live


def _store(backlog, cursor, *, index=None, rendered=None):
    index = walk_index("math-v1", "5Hot", cursor, 50) if index is None else index
    backlog.save(cursor, index, f"q{index}" if rendered is None else rendered,
                 [Generation([104, 105, EOS], [f"stored{cursor}"])])


def test_a_restart_submits_the_stored_backlog_instead_of_generating_it(tmp_path):
    backlog = Backlog(tmp_path, "fp")
    _store(backlog, 1)
    _store(backlog, 2)
    client, generator = _Client(["accepted"] * 3), _WindowGenerator()
    counts = mine_window(job=_job(n=1), hotkey="5Hot", client=client, generator=generator,
                         tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                         window=3, max_steps=1, sleep=lambda s: None, backlog=backlog)
    assert [b["cursor"] for b in client.submitted] == [0, 1, 2]
    assert [b["completions"][0]["proofs"] for b in client.submitted[1:]] == [["stored1"], ["stored2"]]
    assert generator.issued == 1, "only the missing cursor is generated"
    assert counts == {"accepted": 3}
    assert not list(tmp_path.iterdir()), "answered prompts leave the backlog"


def test_the_backlog_trusts_only_its_own_current_prompts(tmp_path):
    _store(Backlog(tmp_path, "old"), 3)
    backlog = Backlog(tmp_path, "fp")
    _store(backlog, 0)
    _store(backlog, 1, index=7)
    _store(backlog, 2, rendered="stale render")
    (tmp_path / "4.json").write_text("{broken")
    client = _Client(["accepted"] * 5)
    client.position = 1
    mine_window(job=_job(n=1), hotkey="5Hot", client=client, generator=_WindowGenerator(),
                tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                window=2, max_steps=4, sleep=lambda s: None, backlog=backlog)
    assert [b["cursor"] for b in client.submitted] == [1, 2, 3, 4]
    assert not any(b["completions"][0]["proofs"][0].startswith("stored") for b in client.submitted)
    assert not list(tmp_path.iterdir())


class _InlineExecutor:
    """Answers every submission before ``submit`` returns, as the route did
    when the miner waited for it: step order is then exact."""

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_finished_prompts_outlive_a_halt_in_the_backlog(tmp_path):
    backlog = Backlog(tmp_path, "fp")
    client = _Client(["miner_banned"])
    with pytest.raises(CorpusMinerHalted):
        mine_window(job=_job(n=1), hotkey="5Hot", client=client, generator=_WindowGenerator(),
                    tokenizer=_Tokenizer(), render=lambda i: f"q{i}", sign=lambda b: "sig",
                    window=3, max_steps=10, sleep=lambda s: None, backlog=backlog,
                    submit_executor=_InlineExecutor())
    # Cursors 2 and 4 finished and waited (4 while cursor 0's answer was taken);
    # cursor 0 was answered, so it is gone.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2.json", "4.json"]
    assert set(Backlog(tmp_path, "fp").load(0)) == {2, 4}


def test_generation_goes_on_while_the_route_answers():
    """A slow answer must not stall the GPU: steps run while it is pending."""
    import threading

    class _SlowClient(_Client):
        def __init__(self, answers):
            super().__init__(answers)
            self.steps_during = []

        def submit(self, body):
            before = generator.steps
            threading.Event().wait(0.2)
            self.steps_during.append(generator.steps - before)
            return super().submit(body)

    class _PacedGenerator(_WindowGenerator):
        def step(self):
            threading.Event().wait(0.005)
            return super().step()

    client, generator = _SlowClient(["accepted"] * 6), _PacedGenerator()
    counts = _mine_window(client, generator, window=3, max_steps=6, n=1)
    assert [b["cursor"] for b in client.submitted] == list(range(6))
    assert counts == {"accepted": 6}
    assert max(client.steps_during) > 0, "the engine kept stepping during a submission"


@pytest.mark.parametrize("reason", ["hotkey_not_registered", "miner_banned"])
def test_a_halting_refusal_stops_the_window(reason):
    client, generator = _Client([reason]), _WindowGenerator()
    with pytest.raises(CorpusMinerHalted, match=reason):
        _mine_window(client, generator, window=3, max_steps=10, n=1)
    assert not generator.live


def test_a_retired_job_ends_the_window_cleanly(caplog):
    import logging

    from reliquary.miner.corpus_miner import CorpusJobRetired

    class _RetiringClient(_Client):
        def submit(self, body):
            self.submitted.append(body)
            raise CorpusJobRetired("410 from the validator")

    client, generator = _RetiringClient([]), _WindowGenerator()
    with caplog.at_level(logging.INFO, logger="reliquary.miner.corpus_miner"):
        counts = _mine_window(client, generator, window=3, max_steps=10, n=1)
    assert counts["job_retired"] == 1 and len(client.submitted) == 1
    assert not generator.live
    assert len([r for r in caplog.records if "retired" in r.getMessage()]) == 1


def _install_fake_vllm(monkeypatch):
    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeGPUModelRunner:
        def execute_model(self, *a, **kw): ...

        def _model_forward(self, *a, **kw): ...

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    fake_gpu_model_runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
    fake_gpu_model_runner_module.GPUModelRunner = _FakeGPUModelRunner

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", fake_gpu_model_runner_module)


def test_the_vllm_generator_stops_only_on_the_jobs_eos(monkeypatch):
    """SamplingParams must stop on the job's eos, not whatever the checkpoint's
    own generation_config lists: vLLM stopping on a DIFFERENT terminator would
    get an honest completion judged (and refused, ``bad_termination``) against
    an eos it never produced."""

    _install_fake_vllm(monkeypatch)

    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    generator = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)

    assert generator._params.stop_token_ids == [EOS]
    assert generator._params.ignore_eos is True
    assert generator._params.min_tokens == 2
    assert generator._params.max_tokens == 64


def test_the_vllm_generator_leaves_the_memory_share_to_vllm_unless_told(monkeypatch):
    """A validator sharing the card needs vLLM to take less than its default
    share; unset, vLLM keeps its own default rather than one we guessed."""
    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)

    default = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)
    shared = VllmGenerator("/fake/checkpoint", sampling, proof, EOS, gpu_memory_utilization=0.5)

    assert "gpu_memory_utilization" not in default._llm.kwargs
    assert shared._llm.kwargs["gpu_memory_utilization"] == 0.5


def test_two_vllm_generators_do_not_share_a_sampling_seed(monkeypatch):
    """vLLM seeds every engine with 0 by default, so two miners sampling the
    same prompt at the same step drew byte-identical completions and the second
    was refused `hash_duplicate` (rehearsal 2026-09-25): each process seeds its
    own engine at random."""
    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)

    seeds = {VllmGenerator("/fake/checkpoint", sampling, proof, EOS)._llm.kwargs.get("seed")
             for _ in range(4)}

    assert None not in seeds and 0 not in seeds
    assert len(seeds) == 4


# --- final review, finding 3: gateway statuses are transient too ---


def _response(status, body=b'{"ok": true}'):
    import httpx

    return httpx.Response(status, content=body,
                          request=httpx.Request("POST", "http://validator/corpus/submit"))


@pytest.mark.parametrize("status", [502, 503, 504])
def test_a_gateway_or_unavailable_status_is_transient(status):
    from reliquary.miner.corpus_miner import issue_corpus_request

    with pytest.raises(CorpusTransientFailure):
        issue_corpus_request(lambda: _response(status, b"{}"))


@pytest.mark.parametrize("status", [400, 404, 422, 500])
def test_other_error_statuses_stay_permanent(status):
    from reliquary.miner.corpus_miner import issue_corpus_request

    with pytest.raises(CorpusPermanentFailure) as caught:
        issue_corpus_request(lambda: _response(status, b'{"detail": "x"}'))
    assert caught.value.status == status


def test_a_transport_error_is_transient_and_a_body_is_returned():
    import httpx

    from reliquary.miner.corpus_miner import issue_corpus_request

    def _fail():
        raise httpx.ConnectError("refused")

    with pytest.raises(CorpusTransientFailure):
        issue_corpus_request(_fail)
    assert issue_corpus_request(lambda: _response(200)) == {"ok": True}


def test_the_vllm_generator_sizes_its_context_to_the_job(monkeypatch):
    """vLLM otherwise reserves the checkpoint's own maximum length (262,144 on
    Qwen3.8-27B), whose KV cache does not fit one H100 next to the weights."""
    from reliquary.miner.corpus_miner import MAX_NUM_SEQS, PROMPT_ALLOWANCE_TOKENS

    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=32768)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    generator = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)
    assert generator._llm.kwargs["max_model_len"] == 32768 + PROMPT_ALLOWANCE_TOKENS
    # vLLM's default of 1,024 concurrent sequences exceeds a hybrid model's
    # Mamba cache on one H100 (320 on Qwen3.8-27B); a step runs only n of them.
    assert generator._llm.kwargs["max_num_seqs"] == MAX_NUM_SEQS


@pytest.mark.parametrize("concurrency, n, window", [(6.79, 1, 6), (6.79, 2, 3), (6.79, 4, 1),
                                                    (0.5, 1, 1), (None, 1, 1), (900.0, 1, 256)])
def test_the_window_holds_every_request_at_full_length(monkeypatch, concurrency, n, window):
    """Rounded down from vLLM's own full-length concurrency: past it the
    scheduler may preempt, and a preempted request's capture is unprovable."""
    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    generator = VllmGenerator("/fake/checkpoint", sampling, SimpleNamespace(chunk_tokens=32, topk=8), EOS)
    generator._llm.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_max_concurrency=concurrency)))
    assert generator.window(n) == window


class _FakeBlockPool:
    def __init__(self, num_gpu_blocks):
        self.num_gpu_blocks = num_gpu_blocks
        self.free = num_gpu_blocks - 1

    def get_num_free_blocks(self):
        return self.free


def _hybrid_kv_generator(monkeypatch, *, num_blocks, reported, **options):
    """Three full-attention groups of 784-token blocks and one Mamba group of a
    single block per request, the layout vLLM gives Qwen3.8-27B."""
    from reliquary.miner.corpus_miner import PROMPT_ALLOWANCE_TOKENS

    class FullAttentionSpec:
        sliding_window = None

        def __init__(self, block_size):
            self.block_size = block_size

    class MambaSpec:
        page_size_bytes = 100

        def max_memory_usage_bytes(self, vllm_config):
            return 100

    _install_fake_vllm(monkeypatch)
    interface = ModuleType("vllm.v1.kv_cache_interface")
    interface.FullAttentionSpec = FullAttentionSpec
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", interface)
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=32768)
    generator = VllmGenerator("/fake/checkpoint", sampling, SimpleNamespace(chunk_tokens=32, topk=8), EOS,
                              **options)
    groups = [SimpleNamespace(kv_cache_spec=FullAttentionSpec(784)) for _ in range(3)]
    groups.append(SimpleNamespace(kv_cache_spec=MambaSpec()))
    finished = []
    generator._llm.llm_engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=32768 + PROMPT_ALLOWANCE_TOKENS),
            cache_config=SimpleNamespace(kv_cache_max_concurrency=reported)),
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=groups, num_blocks=num_blocks),
            waiting=[], kv_cache_manager=SimpleNamespace(block_pool=_FakeBlockPool(num_blocks))))),
        step=lambda: [finished.pop()] if finished else [],
        abort_request=lambda ids, internal: None,
    )
    issued = iter(range(1000))
    generator._llm.enqueue = lambda prompts, params, use_tqdm: [f"{next(issued)}-ab" for _ in prompts]
    generator._kv = generator._kv_layout()
    return generator, finished


def test_room_reserves_each_request_at_its_own_full_length(monkeypatch):
    from reliquary.miner.corpus_miner import MAX_NUM_SEQS

    # 1,225 blocks over 160 per full-length request (3 x 53 + 1): vLLM's 7.66.
    generator, finished = _hybrid_kv_generator(monkeypatch, num_blocks=1225, reported=7.656)
    # A 300-token prompt plus 32,768 tokens (and one async step) holds 3 x 43 + 1.
    assert generator.capacity(300, 1) == 1224 // 130 == 9
    assert generator.capacity(300, 2) == 4
    assert generator.window(1) == MAX_NUM_SEQS
    started = [generator.start([1] * 300, 1) for _ in range(9)]
    assert not generator.room(300, 1)
    finished.append(SimpleNamespace(finished=True, request_id=started[0][0].split("-")[0],
                                    outputs=[SimpleNamespace(token_ids=[EOS])]))
    assert [r for r, _ in generator.step()] == started[0]
    assert generator.room(300, 1)
    generator.start([1] * 300, 1)
    generator.cancel(started[1])
    assert generator.room(300, 1) and not generator.room(8000, 2)


def test_with_a_headroom_prompts_are_admitted_by_the_blocks_actually_free(monkeypatch):
    from reliquary.miner.corpus_miner import OVERCOMMIT_MAX_IN_FLIGHT

    generator, _ = _hybrid_kv_generator(monkeypatch, num_blocks=1225, reported=7.656, kv_headroom=0.1)
    scheduler = generator._llm.llm_engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    assert generator.window(1) == OVERCOMMIT_MAX_IN_FLIGHT
    assert generator.capacity(300, 1) == OVERCOMMIT_MAX_IN_FLIGHT
    for _ in range(12):  # past the 9 a full-length reservation allows
        generator.start([1] * 300, 1)
    # 300 + 4,096 tokens holds 3 x 6 + 1 = 19 blocks; 10% of 1,225 stays spare.
    pool.free = 123 + 19
    assert generator.room(300, 1)
    pool.free = 123 + 18
    assert not generator.room(300, 1)
    pool.free = 1000
    scheduler.waiting.append("preempted or not yet scheduled")
    assert not generator.room(300, 1)


def test_draft_tokens_and_a_headroom_turn_on_positional_capture(monkeypatch):
    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    seen = []
    import reliquary.miner.vllm_hidden_capture as capture_module

    real = capture_module.capture_hidden_states

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(capture_module, "capture_hidden_states", spy)
    plain = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)
    drafted = VllmGenerator("/fake/checkpoint", sampling, proof, EOS, speculative_tokens=2)
    VllmGenerator("/fake/checkpoint", sampling, proof, EOS, kv_headroom=0.1)
    assert [k["positional"] for k in seen] == [False, True, True]
    assert "speculative_config" not in plain._llm.kwargs
    assert drafted._llm.kwargs["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 2}


def test_a_misread_kv_layout_falls_back_to_full_length(monkeypatch):
    generator, _ = _hybrid_kv_generator(monkeypatch, num_blocks=1225, reported=5.0)
    assert generator._kv is None
    assert generator.window(1) == 5 and generator.room(300, 1)


def test_a_vision_checkpoint_is_served_text_only(monkeypatch, tmp_path):
    """The job's prompts are text: a multimodal checkpoint must not reserve its
    vision encoder's profiling memory."""
    import json

    _install_fake_vllm(monkeypatch)
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2, max_new_tokens=64)
    proof = SimpleNamespace(chunk_tokens=32, topk=8)
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": {"depth": 27}}))
    vision = VllmGenerator(str(tmp_path), sampling, proof, EOS)
    text = VllmGenerator("/fake/checkpoint", sampling, proof, EOS)
    assert vision._llm.kwargs["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert "limit_mm_per_prompt" not in text._llm.kwargs


# --------------------------------------------------------------------------
# HttpCorpusClient: the legacy paths, or one job's paths with --job-id
# --------------------------------------------------------------------------


def _validator(jobs):
    """A validator serving ``jobs`` (job_id -> manifest), answering like the real routes."""
    import httpx

    seen = []

    def handle(request):
        seen.append((request.method, request.url.path))
        path = request.url.path
        if path == "/corpus/jobs":
            return httpx.Response(200, json={"jobs": sorted(jobs)})
        if path == "/corpus/job":
            return httpx.Response(200, json=next(iter(jobs.values())))
        if path == "/corpus/cursor/5Hot":
            return httpx.Response(200, json={"hotkey": "5Hot", "cursor": 3})
        if path.startswith("/corpus/jobs/"):
            _, _, _, job_id, what, *rest = path.split("/")
            if job_id not in jobs:
                return httpx.Response(404, json={"detail": "corpus_job_not_served"})
            if what == "job":
                return httpx.Response(200, json=jobs[job_id])
            return httpx.Response(200, json={"hotkey": rest[0], "cursor": 7})
        if path == "/corpus/submit":
            return httpx.Response(200, json={"accepted": True, "reason": "accepted"})
        return httpx.Response(404, json={"detail": "Not Found"})

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://validator"), seen


def test_without_a_job_id_the_client_uses_the_legacy_paths():
    from reliquary.miner.corpus_miner import HttpCorpusClient

    http, seen = _validator({"math": {"job_id": "math"}})
    client = HttpCorpusClient(http)

    assert client.job() == {"job_id": "math"}
    assert client.cursor("5Hot") == 3
    assert client.submit({"job_id": "math"})["accepted"] is True
    assert [p for _, p in seen] == ["/corpus/job", "/corpus/cursor/5Hot", "/corpus/submit"]


def test_with_a_job_id_the_client_uses_that_jobs_paths():
    from reliquary.miner.corpus_miner import HttpCorpusClient

    http, seen = _validator({"math": {"job_id": "math"}, "code": {"job_id": "code"}})
    client = HttpCorpusClient(http, job_id="code")

    assert client.job() == {"job_id": "code"}
    assert client.cursor("5Hot") == 7
    assert client.submit({"job_id": "code"})["accepted"] is True
    assert [p for _, p in seen] == ["/corpus/jobs/code/job", "/corpus/jobs/code/cursor/5Hot",
                                    "/corpus/submit"]


def test_a_multi_job_validator_without_a_job_id_gives_its_first_listed_job():
    from reliquary.miner.corpus_miner import HttpCorpusClient

    http, _ = _validator({"math": {"job_id": "math"}, "code": {"job_id": "code"}})
    client = HttpCorpusClient(http)

    assert client.job() == {"job_id": "math"}
    assert client.served_jobs() == ["code", "math"]


def test_a_job_id_the_validator_does_not_serve_names_the_ones_it_does():
    from reliquary.miner.corpus_miner import CorpusJobSelectionError, HttpCorpusClient

    http, _ = _validator({"math": {"job_id": "math"}, "code": {"job_id": "code"}})

    with pytest.raises(CorpusJobSelectionError) as caught:
        HttpCorpusClient(http, job_id="nope").job()
    assert "nope" in str(caught.value) and "code" in str(caught.value)


def test_corpus_mine_on_a_multi_job_validator_without_job_id_mines_the_default(monkeypatch):
    """Adding a job must not halt a live miner: it keeps mining the validator's
    default job, and is told the others exist."""
    from types import SimpleNamespace

    import bittensor
    import httpx
    import huggingface_hub
    from typer.testing import CliRunner

    import reliquary.protocol.profiles as profiles
    from reliquary.cli.main import app
    from reliquary.protocol.profiles import TASK_CONTRACT_ENV_VAR, TOPLOC_DEPLOYED_DEFAULTS

    class _Downloading(Exception):
        pass

    def download(repo, revision=None):
        raise _Downloading(repo)

    from tests.unit.test_corpus_service import _manifest

    http, seen = _validator({"math": {**_manifest(), "job_id": "math"}, "code": {"job_id": "code"}})
    monkeypatch.setenv(TASK_CONTRACT_ENV_VAR, "/unused-profile-is-patched")
    monkeypatch.setattr(profiles, "ACTIVE_PROTOCOL_PROFILE",
                        SimpleNamespace(profile_id="p", proofs=(TOPLOC_DEPLOYED_DEFAULTS,)))
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(httpx, "Client", lambda **kw: http)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)

    result = CliRunner().invoke(app, ["corpus", "mine", "--validator-url", "http://validator"])

    assert isinstance(result.exception, _Downloading), (result.output, result.exception)
    assert "mining job math" in result.output and "code" in result.output
    assert "--job-id" in result.output


def _single_job_validator():
    """A validator from before several jobs: no /corpus/jobs routes at all."""
    import httpx

    def handle(request):
        if request.url.path == "/corpus/job":
            return httpx.Response(200, json={"job_id": "math"})
        return httpx.Response(404, json={"detail": "Not Found"})

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://validator")


@pytest.mark.parametrize("read", ["job", "contract"])
def test_a_job_id_against_a_single_job_validator_says_to_drop_it(read):
    from reliquary.miner.corpus_miner import CorpusJobSelectionError, HttpCorpusClient

    client = HttpCorpusClient(_single_job_validator(), job_id="math")
    with pytest.raises(CorpusJobSelectionError) as caught:
        getattr(client, read)()
    assert "single job" in str(caught.value) and "--job-id" in str(caught.value)


def test_a_retired_job_is_a_410_job_retired_not_a_permanent_failure():
    from reliquary.miner.corpus_miner import CorpusJobRetired, issue_corpus_request

    with pytest.raises(CorpusJobRetired):
        issue_corpus_request(lambda: _response(410, b'{"detail": "job_retired"}'))


def test_a_retired_job_ends_the_miner_cleanly_with_one_log_line(caplog):
    import logging

    from reliquary.miner.corpus_miner import CorpusJobRetired

    class _RetiringClient(_Client):
        def submit(self, body):
            self.submitted.append(body)
            raise CorpusJobRetired("410 from the validator")

    client = _RetiringClient([])
    sleeps = []
    with caplog.at_level(logging.INFO, logger="reliquary.miner.corpus_miner"):
        counts = mine_steps(job=_job(), hotkey="5Hot", client=client, generator=_Generator(),
                            tokenizer=_Tokenizer(), render=lambda i: f"q{i}",
                            sign=lambda b: "sig", max_steps=10, sleep=sleeps.append)
    assert counts["job_retired"] == 1 and len(client.submitted) == 1 and sleeps == []
    assert len([r for r in caplog.records if "retired" in r.getMessage()]) == 1
