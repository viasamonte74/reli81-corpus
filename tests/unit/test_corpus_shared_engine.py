"""Several corpus jobs mined at once on one generator, and the prompt cache."""

import json
import threading
from types import SimpleNamespace

import pytest

from reliquary.corpus.walk import walk_index
from reliquary.miner.corpus_miner import (
    CorpusMinerHalted,
    Generation,
    JobRun,
    SharedEngine,
    mine_jobs,
    mine_window,
)
from reliquary.miner.prompt_cache import PromptCache, PromptCacheError

EOS = 99


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)


def _job(job_id, n=1, prompt_count=50):
    return SimpleNamespace(job_id=job_id, prompt_count=prompt_count, prompt_start=0,
                           eos_token_id=EOS, checkpoint_sha256="a" * 64,
                           sampling=SimpleNamespace(n=n), prompt_order="miner_walk")


class _Client:
    def __init__(self, answers):
        self.answers = list(answers)
        self.submitted = []
        self.position = 0
        self.lock = threading.Lock()

    def cursor(self, hotkey):
        return self.position

    def submit(self, body):
        with self.lock:
            self.submitted.append(body)
            answer = self.answers.pop(0) if self.answers else "job_complete"
            if answer == "accepted":
                self.position += 1
            return {"reason": answer, "accepted": answer == "accepted"}


class _Engine:
    """One engine for every job: finishes the oldest request each step, and
    records which thread touched it (only the pump may)."""

    def __init__(self):
        self.live = []
        self.issued = 0
        self.threads = set()
        self.cancelled = []
        self.forgotten = []
        self.budgets = {}
        self.room_budgets = set()
        self.lock = threading.Lock()

    def _touch(self):
        self.threads.add(threading.current_thread().name)

    def start(self, prompt_ids, n, **budget):
        self._touch()
        ids = []
        for _ in range(n):
            self.issued += 1
            ids.append(f"{self.issued}-x")
            self.budgets[ids[-1]] = (chr(prompt_ids[0]), budget.get("max_tokens"))
        self.live.extend((i, list(prompt_ids)) for i in ids)
        return ids

    def busy(self):
        self._touch()
        return bool(self.live)

    def step(self):
        self._touch()
        request_id, prompt_ids = self.live.pop(0)
        # The completion echoes its prompt's first token, so a job can tell
        # whether it got its own completion back.
        return [(request_id, [prompt_ids[0], 105, EOS])]

    def room(self, prompt_len, n, **budget):
        self._touch()
        self.room_budgets.add(budget.get("max_tokens"))
        return True

    def rows_for(self, request_id, prompt_len, tokens):
        self._touch()
        return list(tokens)

    def prove(self, rows, tokens):
        return Generation(tokens, ["AAAA"])

    def cancel(self, request_ids):
        self._touch()
        self.cancelled.extend(request_ids)
        self.live = [(r, p) for r, p in self.live if r not in request_ids]

    def forget(self, request_ids):
        self.forgotten.extend(request_ids)

    def window(self, n):
        return 4


def _run(name, client, prefix, max_steps=4, n=1, **extra):
    return JobRun(name=name, kwargs=dict(
        job=_job(name, n=n), hotkey="5Hot", client=client, tokenizer=_Tokenizer(),
        render=lambda i, p=prefix: f"{p}{i}", sign=lambda b: "sig", window=3,
        max_steps=max_steps, sleep=lambda s: None, **extra))


def test_two_jobs_on_one_engine_each_submit_their_own_walk_in_order():
    math, logic = _Client(["accepted"] * 4), _Client(["accepted"] * 4)
    engine = _Engine()
    results = mine_jobs([_run("math", math, "m"), _run("logic", logic, "l", n=2)], engine)
    assert results == {"math": {"accepted": 4}, "logic": {"accepted": 4}}
    for client, name in ((math, "math"), (logic, "logic")):
        assert [b["cursor"] for b in client.submitted] == [0, 1, 2, 3]
        assert [b["prompt_index"] for b in client.submitted] == [
            walk_index(name, "5Hot", c, 50) for c in range(4)]
    # Every completion went back to the job whose prompt it continued.
    assert all(c["tokens"][0] == ord("m") for b in math.submitted for c in b["completions"])
    assert all(len(b["completions"]) == 2 and c["tokens"][0] == ord("l")
               for b in logic.submitted for c in b["completions"])
    assert engine.threads == {"engine"}, "only the pump thread may touch the engine"
    assert not engine.live


def test_each_job_generates_under_its_own_completion_budget():
    engine = _Engine()
    short = _run("if", _Client(["accepted"] * 3), "i", max_steps=3)
    short.max_tokens = 8192
    mine_jobs([_run("math", _Client(["accepted"] * 3), "m", max_steps=3), short], engine)
    assert {budget for prefix, budget in engine.budgets.values() if prefix == "m"} == {None}
    assert {budget for prefix, budget in engine.budgets.values() if prefix == "i"} == {8192}
    assert engine.room_budgets <= {None, 8192}


def test_one_job_ending_leaves_the_other_mining():
    done, going = _Client(["job_complete"]), _Client(["accepted"] * 5)
    results = mine_jobs([_run("done", done, "d", max_steps=10),
                         _run("going", going, "g", max_steps=5)], _Engine())
    assert results["done"] == {"job_complete": 1}
    assert results["going"] == {"accepted": 5}


def test_a_hotkey_refusal_in_one_job_stops_every_job():
    banned = _Client(["miner_banned"])
    slow = _Client(["accepted"] * 1000)
    with pytest.raises(CorpusMinerHalted, match="miner_banned"):
        mine_jobs([_run("banned", banned, "b", max_steps=10),
                   _run("slow", slow, "s", max_steps=None)], _Engine())


def test_a_cancelled_request_already_finished_has_its_rows_forgotten():
    engine = _Engine()
    shared = SharedEngine(engine)
    try:
        view = shared.view()
        first, = view.start([1, 2], 1)
        deadline = threading.Event()
        while view._inbox.empty() and not deadline.wait(0.01):
            pass
        second, = view.start([3, 4], 1)
        view.cancel([first, second])
        assert first in engine.forgotten, "finished, so only its rows remain"
        assert second in engine.cancelled or second in engine.forgotten
        assert not view._pending
    finally:
        shared.stop()


def test_consecutive_prompt_mismatches_end_the_job():
    client = _Client(["prompt_mismatch"] * 5)
    counts = mine_window(job=_job("logic"), hotkey="5Hot", client=client,
                         generator=SharedEngine(_Engine()).view(), tokenizer=_Tokenizer(),
                         render=lambda i: f"q{i}", sign=lambda b: "sig", window=2,
                         max_steps=None, sleep=lambda s: None)
    assert counts["prompt_mismatch"] == 3 and counts["stopped_prompt_mismatch"] == 1
    assert len(client.submitted) == 3


def test_a_mismatch_between_accepts_does_not_end_the_job():
    client = _Client(["prompt_mismatch", "prompt_mismatch", "accepted"] * 2 + ["accepted"])
    counts = mine_window(job=_job("logic"), hotkey="5Hot", client=client,
                         generator=SharedEngine(_Engine()).view(), tokenizer=_Tokenizer(),
                         render=lambda i: f"q{i}", sign=lambda b: "sig", window=2,
                         max_steps=3, sleep=lambda s: None)
    assert "stopped_prompt_mismatch" not in counts and counts["accepted"] == 3


def _cache(tmp_path, job, rows, **header):
    path = tmp_path / "prompts.jsonl"
    head = {"schema": "reliquary/corpus-prompt-cache/v1", "job_id": job.job_id,
            "prompt_source": job.prompt_source, "prompt_start": job.prompt_start,
            "prompt_count": job.prompt_count, "profile_id": "corpus-logic-v1",
            "environment_manifest_sha256": "f" * 64, **header}
    path.write_text("\n".join([json.dumps(head)] + [json.dumps(r) for r in rows]) + "\n")
    return path


def _cache_job(**fields):
    return SimpleNamespace(**{"job_id": "logic-v1", "prompt_source": "reliquary_logic_v2",
                              "prompt_start": 100, "prompt_count": 3, **fields})


def test_the_prompt_cache_answers_by_source_index(tmp_path):
    job = _cache_job()
    cache = PromptCache(_cache(tmp_path, job, ["a", "b", "c"]), job)
    assert [cache.prompt(i) for i in (100, 101, 102)] == ["a", "b", "c"]
    with pytest.raises(PromptCacheError):
        cache.prompt(103)


@pytest.mark.parametrize("field,value", [("job_id", "math-v1"), ("prompt_start", 0),
                                         ("prompt_count", 4), ("prompt_source", "other")])
def test_a_prompt_cache_for_other_rows_is_refused(tmp_path, field, value):
    job = _cache_job()
    path = _cache(tmp_path, job, ["a", "b", "c"], **{field: value})
    with pytest.raises(PromptCacheError, match=field):
        PromptCache(path, job)


def test_a_truncated_prompt_cache_is_refused(tmp_path):
    job = _cache_job()
    with pytest.raises(PromptCacheError, match="rows"):
        PromptCache(_cache(tmp_path, job, ["a", "b"]), job)
