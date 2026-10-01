"""`corpus mine --extra-job`: which jobs one engine may serve beside the main one."""

from __future__ import annotations

import copy
import json

import httpx
import pytest
import typer

from reliquary.cli.main import _extra_corpus_job, _parse_prompt_caches
from reliquary.corpus.job import parse_job
from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, TOPLOC_DEPLOYED_DEFAULTS

MATH = {
    "schema": "reliquary/corpus-job/v1", "job_id": "math-v1",
    "checkpoint_repo": "Qwen/Qwen3.8-27B", "checkpoint_revision": "1d4bf0f2",
    "checkpoint_sha256": "3e" * 32, "eos_token_id": 248046,
    "filter": {"grader_id": "math", "threshold": 1.0},
    "prompt_count": 10, "prompt_order": "miner_walk", "prompt_source": "math",
    "prompt_start": 0, "renderer_id": "chat-template-thinking-v1",
    "sampling": {"max_new_tokens": 32768, "min_new_tokens": 16, "n": 8,
                 "temperature": 1.0, "top_k": 20, "top_p": 0.95},
    "slots_per_prompt": 1, "deadline_round": None,
}
LOGIC = {**MATH, "job_id": "logic-v1", "prompt_source": "reliquary_logic_v2",
         "prompt_start": 100, "prompt_count": 3,
         "filter": {"grader_id": "reliquary_logic_v2", "threshold": 1.0},
         "sampling": {**MATH["sampling"], "n": 2}}


def _contract(profile_id="corpus-logic-v1", **proof):
    contract = ACTIVE_PROTOCOL_PROFILE.to_generation_contract()
    contract["profile_id"] = profile_id
    contract["proofs"] = [{**TOPLOC_DEPLOYED_DEFAULTS.to_contract(), **proof}]
    return contract


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        (message,) = messages
        return f"<user>{message['content']}</user><think={enable_thinking}>"


def _http(job, contract, failures=()):
    failures = list(failures)

    def handle(request):
        if failures:
            failure = failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, json={"detail": "busy"})
        if request.url.path == f"/corpus/jobs/{job['job_id']}/job":
            return httpx.Response(200, json=job)
        if request.url.path == f"/corpus/jobs/{job['job_id']}/contract":
            return httpx.Response(200, json=contract)
        return httpx.Response(404, json={"detail": "corpus_job_not_served"})

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://v")


def _cache(tmp_path, job, rows=("a", "b", "c"), profile_id="corpus-logic-v1"):
    path = tmp_path / "prompts.jsonl"
    head = {"schema": "reliquary/corpus-prompt-cache/v1", "job_id": job["job_id"],
            "prompt_source": job["prompt_source"], "prompt_start": job["prompt_start"],
            "prompt_count": job["prompt_count"], "profile_id": profile_id,
            "environment_manifest_sha256": "f" * 64}
    path.write_text("\n".join([json.dumps(head)] + [json.dumps(r) for r in rows]) + "\n")
    return str(path)


def _extra(job=LOGIC, contract=None, cache_path=None):
    return _extra_corpus_job(
        _http(job, contract or _contract()), job["job_id"], primary=parse_job(MATH),
        proof=TOPLOC_DEPLOYED_DEFAULTS, tokenizer=_Tokenizer(), encode=None,
        cache_path=cache_path)


def test_a_compatible_job_renders_its_cached_rows_through_the_chat_template(tmp_path):
    job, client, render, profile_id = _extra(cache_path=_cache(tmp_path, LOGIC))
    assert job.job_id == "logic-v1" and job.sampling.n == 2
    assert profile_id == "corpus-logic-v1"
    assert render(101) == "<user>b</user><think=True>"
    assert client.contract()["profile_id"] == "corpus-logic-v1"


def _changed(path, value):
    job = copy.deepcopy(LOGIC)
    *parents, leaf = path
    target = job
    for key in parents:
        target = target[key]
    target[leaf] = value
    return job


@pytest.mark.parametrize("path,value", [
    (("checkpoint_revision",), "other"),
    (("checkpoint_sha256",), "ab" * 32),
    (("eos_token_id",), 1),
    (("sampling", "top_p"), 0.9),
    (("sampling", "max_new_tokens"), 65536),
    (("sampling", "min_new_tokens"), 32),
    (("sampling", "temperature"), 0.6),
])
def test_a_job_one_engine_cannot_serve_is_refused(tmp_path, capsys, path, value):
    job = _changed(path, value)
    with pytest.raises(typer.Exit) as exc:
        _extra(job=job, cache_path=_cache(tmp_path, job))
    assert exc.value.exit_code == 2
    assert path[-1] in capsys.readouterr().err


def test_a_shorter_completion_budget_is_served_on_the_same_engine(tmp_path):
    job = {**_changed(("sampling", "max_new_tokens"), 8192), "renderer_id": "chat-template-v1"}
    extra, _, render, _ = _extra(job=job, cache_path=_cache(tmp_path, job))
    assert extra.sampling.max_new_tokens == 8192
    assert render(100) == "<user>a</user><think=False>"


def test_a_job_proved_differently_is_refused(tmp_path, capsys):
    with pytest.raises(typer.Exit):
        _extra(contract=_contract(chunk_tokens=64), cache_path=_cache(tmp_path, LOGIC))
    assert "proves completions differently" in capsys.readouterr().err


def test_a_prompt_cache_rendered_under_another_profile_is_refused(tmp_path, capsys):
    with pytest.raises(typer.Exit):
        _extra(cache_path=_cache(tmp_path, LOGIC, profile_id="corpus-logic-v0"))
    assert "corpus-logic-v0" in capsys.readouterr().err


def test_a_source_the_active_contract_lacks_needs_a_prompt_cache(capsys):
    assert "reliquary_logic_v2" not in ACTIVE_PROTOCOL_PROFILE.environments
    with pytest.raises(typer.Exit):
        _extra()
    assert "--prompt-cache" in capsys.readouterr().err


def test_the_main_job_cannot_be_its_own_extra(tmp_path, capsys):
    with pytest.raises(typer.Exit):
        _extra(job=MATH, contract=_contract("corpus-math-v1"))
    assert "--job-id job" in capsys.readouterr().err


def test_an_unserved_extra_job_is_refused(capsys):
    client = _http(LOGIC, _contract())
    with pytest.raises(typer.Exit):
        _extra_corpus_job(client, "nope", primary=parse_job(MATH),
                          proof=TOPLOC_DEPLOYED_DEFAULTS, tokenizer=_Tokenizer(),
                          encode=None, cache_path=None)
    assert "nope" in capsys.readouterr().err


def _fetching(failures, tmp_path, attempts=8):
    slept = []
    result = _extra_corpus_job(
        _http(LOGIC, _contract(), failures), "logic-v1", primary=parse_job(MATH),
        proof=TOPLOC_DEPLOYED_DEFAULTS, tokenizer=_Tokenizer(), encode=None,
        cache_path=_cache(tmp_path, LOGIC), fetch_attempts=attempts, sleep=slept.append)
    return result, slept


def test_a_busy_validator_is_waited_for(tmp_path):
    (job, *_), slept = _fetching(
        [503, httpx.ReadTimeout("timed out"), 504, httpx.ConnectError("refused")], tmp_path)
    assert job.job_id == "logic-v1" and slept == [5, 10, 20, 40]


def test_a_validator_busy_past_every_attempt_refuses_the_job(tmp_path, capsys):
    with pytest.raises(typer.Exit):
        _fetching([503] * 3, tmp_path, attempts=3)
    assert "did not serve it" in capsys.readouterr().err


def test_a_client_error_is_not_retried(tmp_path, capsys):
    with pytest.raises(typer.Exit):
        _fetching([400, 503], tmp_path)
    assert "400" in capsys.readouterr().err


def test_prompt_caches_are_named_by_job():
    assert _parse_prompt_caches(["logic-v1=/a/b.jsonl", "code-v1=c=d"]) == {
        "logic-v1": "/a/b.jsonl", "code-v1": "c=d"}
    assert _parse_prompt_caches(None) == {}
    for bad in ("logic-v1", "=x", "logic-v1="):
        with pytest.raises(typer.BadParameter):
            _parse_prompt_caches([bad])
