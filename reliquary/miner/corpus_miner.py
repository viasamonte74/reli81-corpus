"""A corpus miner: walk the job's prompts in this hotkey's order, generate with
the job's sampling, prove every completion from its own decode activations,
sign, submit.

The loop is written against three small seams (generator, client, signer) so
it is tested without a GPU; ``VllmGenerator`` is the real generator. A
``CorpusClient`` may raise ``CorpusTransientFailure`` (502/503/504, a
timeout, a transport error) -- the signed body is idempotent, so the loop
retries the identical submission rather than paying for a fresh generation --
or ``CorpusPermanentFailure`` (any other HTTP error, or a body this client
cannot make sense of); a run of ``max_consecutive_failures`` of the latter
raises ``CorpusMinerHalted`` rather than spinning forever on a route that
will never answer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import secrets
import time
from typing import Protocol

from reliquary.corpus.encoding import completion_text, prompt_token_ids
from reliquary.corpus.walk import job_walk_index

logger = logging.getLogger(__name__)

# Reasons after which the cursor on the validator is the truth, not ours.
_RESYNC = frozenset({"prompt_full", "bad_cursor", "prompt_mismatch"})
# Refusals no retry can change: generating on would only burn the card.
_HALT = frozenset({"hotkey_not_registered", "miner_banned"})
# Skip refusals that only say the read was stale: read `next` again.
_SKIP_REREAD = frozenset({"bad_cursor", "prompt_not_full"})
# How many stale skips in a row before generating anyway; submit then settles it.
_MAX_STALE_SKIPS = 3

# Backoff delays for a retried request, in seconds; the last value repeats.
# Bounded so a long outage does not turn into an ever-growing sleep.
_TRANSIENT_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)

# How many consecutive permanent failures (or reason-less answers) the loop
# tolerates on one call before giving up on it entirely.
_MAX_CONSECUTIVE_FAILURES = 5

# Statuses that say "not now" rather than "never": ledger contention or a store
# outage (503), and a proxy in front of the validator timing out or losing it
# (502/504). A short outage must not count toward the permanent-failure halt.
TRANSIENT_STATUSES = frozenset({502, 503, 504})


class CorpusTransientFailure(Exception):
    """Ledger contention (HTTP 503) or a transport-level failure (timeout,
    connection error). The request that failed is idempotent -- it is either
    a cursor read or a signed, already-built submission -- so the caller
    retries the SAME request rather than building a new one."""


class CorpusPermanentFailure(Exception):
    """An HTTP error this client has no reason to expect will clear on retry
    (404/422/500/...), or a response this client could not parse. Counted by
    the loop rather than left to crash it outright."""

    def __init__(self, message: str, *, status: int | None = None, detail=None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class CorpusJobRetired(Exception):
    """The validator answered 410 ``job_retired``: the job admits nothing more.
    A job end, not a failure: the miner stops it without retrying."""


class CorpusMinerHalted(Exception):
    """Raised out of ``mine_steps`` after too many consecutive permanent
    failures on one call, so the CLI can report why and exit non-zero
    instead of the process looping on a job or route that will never
    answer."""

    def __init__(self, message: str, *, counts: dict[str, int]) -> None:
        super().__init__(message)
        self.counts = dict(counts)


def _error_detail(response):
    try:
        return response.json()
    except ValueError:
        return response.text[:500]


def _error_object(response) -> dict:
    detail = _error_detail(response)
    return detail if isinstance(detail, dict) else {}


def issue_corpus_request(request_call):
    """Run one httpx request against the corpus validator, translating its
    outcome into the two exceptions ``mine_steps`` understands. A status in
    ``TRANSIENT_STATUSES`` and a transport failure (timeout, connection error)
    are transient -- the caller retries the SAME idempotent request; every other
    error status, or a body this client cannot parse as JSON, is permanent."""
    import httpx

    try:
        response = request_call()
    except httpx.TransportError as exc:
        raise CorpusTransientFailure(f"transport error: {exc}") from exc
    if response.status_code == 410 and _error_object(response).get("detail") == "job_retired":
        raise CorpusJobRetired(f"410 job_retired from {response.request.url}")
    if response.status_code in TRANSIENT_STATUSES:
        raise CorpusTransientFailure(f"{response.status_code} from {response.request.url}")
    if response.status_code >= 400:
        raise CorpusPermanentFailure(
            f"{response.status_code} from {response.request.url}",
            status=response.status_code,
            detail=_error_detail(response),
        )
    try:
        return response.json()
    except ValueError as exc:
        raise CorpusPermanentFailure(
            f"non-JSON body from {response.request.url}: {exc}",
            status=response.status_code,
        ) from exc


class CorpusJobSelectionError(Exception):
    """This miner named a job the validator does not serve: pass ``--job-id``
    with one of the listed jobs."""


class HttpCorpusClient:
    """The ``CorpusClient`` over HTTP: the legacy paths (the validator's default
    job), or with ``job_id`` that job's own paths on a validator serving several."""

    def __init__(self, http, *, job_id: str | None = None) -> None:
        self._http = http
        self._job_id = job_id
        self.scoped_submit = False

    def served_jobs(self) -> list:
        """Every job the validator serves; empty when it cannot say."""
        try:
            return list(self._http.get("/corpus/jobs").json()["jobs"])
        except Exception:
            return []

    def _refuse_unserved(self, response) -> None:
        """A job-scoped 404: a job this validator does not serve, or a validator
        from before several jobs, which has no job-scoped routes at all."""
        if response.status_code != 404:
            return
        if _error_object(response).get("detail") == "corpus_job_not_served":
            raise CorpusJobSelectionError(
                f"the validator does not serve job {self._job_id!r}; it serves {self.served_jobs()}"
            )
        raise CorpusJobSelectionError(
            "this validator serves a single job and has no job-scoped routes; "
            "drop --job-id, or ask its operator to update it"
        )

    def job(self) -> dict:
        if self._job_id is None:
            response = self._http.get("/corpus/job")
        else:
            response = self._http.get(f"/corpus/jobs/{self._job_id}/job")
            self._refuse_unserved(response)
        response.raise_for_status()
        return response.json()

    def contract(self) -> dict:
        """The task contract the validator serves for this job (or its only one)."""
        if self._job_id is None:
            response = self._http.get("/corpus/contract")
        else:
            response = self._http.get(f"/corpus/jobs/{self._job_id}/contract")
            self._refuse_unserved(response)
        response.raise_for_status()
        return response.json()

    def cursor(self, hotkey: str) -> int:
        path = (f"/corpus/cursor/{hotkey}" if self._job_id is None
                else f"/corpus/jobs/{self._job_id}/cursor/{hotkey}")
        return int(issue_corpus_request(lambda: self._http.get(path))["cursor"])

    def submit(self, body: dict) -> dict:
        # An eval job is served behind its own prefix only (the eval control):
        # set by the caller once the job's manifest names an eval set.
        path = (f"/corpus/jobs/{self._job_id}/submit"
                if self._job_id is not None and self.scoped_submit else "/corpus/submit")
        return issue_corpus_request(lambda: self._http.post(path, json=body))

    def eval_prompts(self) -> bytes:
        """An eval job's prompt lines, which the job's manifest hashes."""
        response = self._http.get(f"/corpus/jobs/{self._job_id}/eval-prompts")
        self._refuse_unserved(response)
        response.raise_for_status()
        return response.content

    def _path(self, tail: str) -> str:
        return (f"/corpus/{tail}" if self._job_id is None
                else f"/corpus/jobs/{self._job_id}/{tail}")

    def next_prompt(self, hotkey: str) -> dict | None:
        """Where this hotkey's walk stands and the slots left there, or None
        from a validator that predates the route (404) or a job that has no
        walk (409)."""
        return _unless_absent(lambda: self._http.get(self._path(f"next/{hotkey}")))

    def skip(self, body: dict) -> dict | None:
        """Step over a full prompt, or None from a validator without the route."""
        return _unless_absent(lambda: self._http.post(self._path("skip"), json=body))


def _unless_absent(request_call):
    try:
        return issue_corpus_request(request_call)
    except CorpusPermanentFailure as exc:
        # 404: a validator from before the routes. 409: a job with no walk.
        if exc.status in (404, 409):
            return None
        raise


@dataclass(frozen=True)
class Generation:
    tokens: list[int]
    proofs: list[str]


class Generator(Protocol):
    def generate(self, prompt_ids: list[int], n: int) -> list[Generation]: ...


class CorpusClient(Protocol):
    def job(self) -> dict: ...

    def cursor(self, hotkey: str) -> int: ...

    def submit(self, body: dict) -> dict: ...


def build_submission(*, job, hotkey, cursor, prompt_index, rendered_prompt, generations,
                     tokenizer, sign) -> dict:
    body = {
        "job_id": job.job_id,
        "miner_hotkey": hotkey,
        "cursor": cursor,
        "prompt_index": prompt_index,
        "checkpoint_sha256": job.checkpoint_sha256,
        "rendered_prompt": rendered_prompt,
        "completions": [
            {"tokens": list(g.tokens), "text": completion_text(tokenizer, g.tokens, job.eos_token_id),
             "proofs": list(g.proofs)}
            for g in generations
        ],
        "signature": "",
    }
    body["signature"] = sign(body)
    return body


def build_skip(*, job, hotkey, cursor, prompt_index, to_cursor, sign) -> dict:
    body = {
        "job_id": job.job_id,
        "miner_hotkey": hotkey,
        "cursor": cursor,
        "prompt_index": prompt_index,
        "to_cursor": to_cursor,
        "signature": "",
    }
    body["signature"] = sign(body)
    return body


def _retry(call, *, sleep, counts, max_consecutive_failures):
    """Run ``call`` (a zero-argument callable bound to one idempotent
    request), retrying on failure.

    A transient failure always retries with bounded exponential backoff --
    nothing is lost by trying the same request again. A permanent failure is
    counted, and after ``max_consecutive_failures`` in a row this call is not
    worth retrying further: raises ``CorpusMinerHalted``. The count resets
    whenever a transient failure or a success intervenes, so it measures a
    genuine consecutive run against this one call.
    """
    attempt = 0
    consecutive_permanent = 0
    while True:
        try:
            return call()
        except CorpusTransientFailure as exc:
            consecutive_permanent = 0
            delay = _TRANSIENT_BACKOFF_SECONDS[min(attempt, len(_TRANSIENT_BACKOFF_SECONDS) - 1)]
            logger.warning("corpus request hit a transient failure: %s; retrying in %.0fs", exc, delay)
            sleep(delay)
            attempt += 1
        except CorpusPermanentFailure as exc:
            consecutive_permanent += 1
            counts["permanent_failure"] += 1
            logger.error("corpus request failed (status=%s): %s", exc.status, exc.detail or exc)
            if consecutive_permanent >= max_consecutive_failures:
                raise CorpusMinerHalted(
                    f"{consecutive_permanent} consecutive permanent failures: {exc}",
                    counts=counts,
                ) from exc
            delay = _TRANSIENT_BACKOFF_SECONDS[min(attempt, len(_TRANSIENT_BACKOFF_SECONDS) - 1)]
            sleep(delay)
            attempt += 1


def _can_skip(job, client, sign_skip) -> bool:
    return (
        sign_skip is not None
        and getattr(job, "prompt_order", None) == "miner_walk"
        and callable(getattr(client, "next_prompt", None))
        and callable(getattr(client, "skip", None))
    )


def _past_full_prompts(*, job, hotkey, client, cursor, sign_skip, counts, retry_kwargs):
    """The cursor of the next prompt worth generating for (None when the job
    is complete), and whether skipping still works against this validator."""
    stale = 0
    while True:
        position = _retry(lambda: client.next_prompt(hotkey), **retry_kwargs)
        if position is None:
            logger.info("the validator has no next/skip routes: generating for every step")
            return cursor, False
        try:
            cursor = int(position["cursor"])
            index = int(position["prompt_index"])
            remaining = int(position["slots_remaining"])
            skip_to = int(position["skip_to"])
            if skip_to <= cursor:
                raise ValueError(f"skip_to {skip_to} is not past cursor {cursor}")
        except (KeyError, TypeError, ValueError):
            logger.warning("unusable next answer %r: generating for every step", position)
            return cursor, False
        if remaining > 0:
            return cursor, True
        if index != job_walk_index(job, hotkey, cursor):
            # Not our walk: generating lets submit name the disagreement.
            logger.warning("the validator's walk names prompt %d at cursor %d, ours %d",
                           index, cursor, job_walk_index(job, hotkey, cursor))
            return cursor, True
        # One skip over the whole run of full prompts `next` found.
        body = build_skip(job=job, hotkey=hotkey, cursor=cursor, prompt_index=index,
                          to_cursor=skip_to, sign=sign_skip)
        answer = _retry(lambda: client.skip(body), **retry_kwargs)
        if answer is None:
            logger.info("the validator has no skip route: generating for every step")
            return cursor, False
        reason = str(answer.get("reason"))
        if answer.get("skipped"):
            counts["skipped"] += 1
            stale = 0
            continue
        if reason == "job_complete":
            counts["job_complete"] += 1
            return None, True
        counts[f"skip_{reason}"] += 1
        if reason in _HALT:
            raise CorpusMinerHalted(f"the validator refused this hotkey: {reason}",
                                    counts=dict(counts))
        if reason in _SKIP_REREAD and stale < _MAX_STALE_SKIPS:
            stale += 1
            continue
        if reason not in _SKIP_REREAD:
            logger.warning("corpus skip refused: %s %s; generating for every step",
                           reason, answer.get("detail"))
            return cursor, False
        return cursor, True


def mine_steps(*, job, hotkey, client, generator, tokenizer, render, sign,
               max_steps: int | None = None, sleep=time.sleep,
               max_consecutive_failures: int = _MAX_CONSECUTIVE_FAILURES,
               sign_skip=None) -> dict[str, int]:
    """``_mine_steps``, ended cleanly (one log line, no retry) when the
    validator says the job is retired."""
    counts: Counter[str] = Counter()
    try:
        return _mine_steps(job=job, hotkey=hotkey, client=client, generator=generator,
                           tokenizer=tokenizer, render=render, sign=sign, max_steps=max_steps,
                           sleep=sleep, max_consecutive_failures=max_consecutive_failures,
                           sign_skip=sign_skip, counts=counts)
    except CorpusJobRetired as exc:
        counts["job_retired"] += 1
        logger.info("corpus job %s is retired; stopping it (%s)", job.job_id, exc)
        return dict(counts)


def _mine_steps(*, job, hotkey, client, generator, tokenizer, render, sign,
                max_steps: int | None, sleep, max_consecutive_failures: int,
                sign_skip, counts: Counter) -> dict[str, int]:
    """Mine up to ``max_steps`` generations.

    With ``sign_skip`` on a ``miner_walk`` job, each step first asks the
    validator where the walk stands and skips full prompts with a signed skip
    rather than generating for them. A validator without those routes, or one
    that cannot verify a skip, is mined exactly as before.
    """
    retry_kwargs = dict(sleep=sleep, counts=counts, max_consecutive_failures=max_consecutive_failures)
    cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
    steps = 0
    consecutive_unreasoned = 0
    skipping = _can_skip(job, client, sign_skip)

    while max_steps is None or steps < max_steps:
        steps += 1
        if skipping:
            cursor, skipping = _past_full_prompts(
                job=job, hotkey=hotkey, client=client, cursor=cursor, sign_skip=sign_skip,
                counts=counts, retry_kwargs=retry_kwargs,
            )
            if cursor is None:
                break
        # A SOURCE index: the one `render` draws and the route renders again.
        prompt_index = job_walk_index(job, hotkey, cursor)
        rendered = render(prompt_index)
        try:
            generations = generator.generate(prompt_token_ids(tokenizer, rendered), job.sampling.n)
        except ValueError as exc:
            # E.g. a row-count mismatch out of `completion_rows`: vLLM
            # preempted and recomputed a request mid-batch under KV
            # pressure, so this step's activations cannot be trusted. The
            # generator has already dropped its own captured rows for the
            # request ids in this step; there is nothing here to submit, so
            # resync the cursor (this step consumed nothing) and move on.
            logger.warning("dropping a corpus generation step: %s", exc)
            counts["generation_failed"] += 1
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
            continue
        body = build_submission(
            job=job, hotkey=hotkey, cursor=cursor, prompt_index=prompt_index,
            rendered_prompt=rendered, generations=generations, tokenizer=tokenizer, sign=sign,
        )
        answer = _retry(lambda: client.submit(body), **retry_kwargs)
        reason = answer.get("reason")
        if reason is None:
            # A response this route never sends without one: count it the
            # same way a permanent HTTP failure is counted, rather than loop
            # forever on an answer nothing here can act on.
            consecutive_unreasoned += 1
            counts["permanent_failure"] += 1
            logger.error("corpus submission answered with no reason: %r", answer)
            if consecutive_unreasoned >= max_consecutive_failures:
                raise CorpusMinerHalted(
                    f"{consecutive_unreasoned} consecutive corpus answers carried no reason",
                    counts=dict(counts),
                )
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
            continue
        consecutive_unreasoned = 0
        reason = str(reason)
        counts[reason] += 1
        if reason == "job_complete":
            break
        if reason in _HALT:
            raise CorpusMinerHalted(f"the validator refused this hotkey: {reason}",
                                    counts=dict(counts))
        if answer.get("accepted"):
            cursor += 1
        elif reason in _RESYNC:
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
        else:
            logger.warning("corpus submission refused: %s %s", reason, answer.get("detail"))
            cursor = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
    return dict(counts)


@dataclass
class _Slot:
    prompt_index: int
    rendered: str
    prompt_ids: list[int]
    request_ids: list[str]
    generations: dict


class Backlog:
    """Finished prompts waiting for their cursor, kept on disk so a restart
    submits them instead of generating them again.

    One file per cursor, written whole (temp file, then rename) once every
    completion of the prompt is proved, and removed once the route answers it.
    A file is trusted only under the fingerprint it was written with (job,
    proof parameters, protocol profile) and only for the prompt the walk still
    puts at its cursor.
    """

    def __init__(self, directory, fingerprint: str) -> None:
        from pathlib import Path

        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fingerprint = fingerprint

    def _path(self, cursor: int):
        return self._dir / f"{cursor}.json"

    def save(self, cursor: int, prompt_index: int, rendered: str, generations: list[Generation]) -> None:
        import json
        import os

        record = {"fingerprint": self._fingerprint, "cursor": cursor, "prompt_index": prompt_index,
                  "rendered": rendered,
                  "generations": [{"tokens": g.tokens, "proofs": g.proofs} for g in generations]}
        path = self._path(cursor)
        temp = path.with_suffix(".tmp")
        try:
            temp.write_text(json.dumps(record, separators=(",", ":")))
            os.replace(temp, path)
        except OSError as exc:
            # Only a restart needs the file: keep mining without it.
            logger.warning("cannot store cursor %d in the backlog: %s", cursor, exc)

    def discard(self, cursor: int) -> None:
        self._path(cursor).unlink(missing_ok=True)

    def load(self, ledger: int) -> dict[int, tuple[int, str, list[Generation]]]:
        """Stored prompts at or past ``ledger``; everything else is removed."""
        import json

        for temp in self._dir.glob("*.tmp"):
            temp.unlink(missing_ok=True)
        stored = {}
        for path in self._dir.glob("*.json"):
            try:
                record = json.loads(path.read_text())
                cursor = int(record["cursor"])
                keep = (record.get("fingerprint") == self._fingerprint and cursor >= ledger
                        and path.stem == str(cursor))
                if keep:
                    stored[cursor] = (int(record["prompt_index"]), str(record["rendered"]),
                                      [Generation(list(g["tokens"]), list(g["proofs"]))
                                       for g in record["generations"]])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("ignoring unreadable backlog file %s: %s", path.name, exc)
                keep = False
            if not keep:
                path.unlink(missing_ok=True)
        return stored


class WindowGenerator(Protocol):
    def start(self, prompt_ids: list[int], n: int) -> list[str]: ...
    def step(self) -> list[tuple[str, list[int]]]: ...
    def finish(self, request_id: str, prompt_len: int, tokens: list[int]) -> Generation: ...
    def cancel(self, request_ids: list[str]) -> None: ...
    def busy(self) -> bool: ...


def mine_window(*, job, hotkey, client, generator, tokenizer, render, sign, window: int,
                max_steps: int | None = None, sleep=time.sleep,
                max_consecutive_failures: int = _MAX_CONSECUTIVE_FAILURES,
                sign_skip=None, max_lookahead: int = 256,
                backlog: Backlog | None = None, submit_executor=None,
                max_prompt_mismatches: int = 3) -> dict[str, int]:
    """Mine like ``mine_steps``, with up to ``window`` prompts generating at once.

    The route accepts a hotkey's submission only at its ledger cursor, so the
    window generates the next cursors of the walk and submits each prompt as
    soon as every cursor before it is in; a prompt that finishes early waits,
    and its place in the window is refilled at once. Every request must stay
    inside the KV cache at its full length, or vLLM preempts and recomputes it
    and its capture cannot be proved: a generator with ``room`` admits each
    prompt by its own length, otherwise ``window`` alone must guarantee it
    (``VllmGenerator.window``).

    Full prompts are skipped only when nothing is generating ahead of the
    ledger cursor. A lookahead prompt that fills meanwhile is refused
    ``prompt_full``, which moves the ledger past it as a skip would.

    With a ``backlog``, every finished prompt is stored until the route answers
    it, and a restart resumes with the stored prompts still ahead of the ledger.

    ``max_prompt_mismatches`` consecutive ``prompt_mismatch`` answers end the
    job: the ledger stays on a refused prompt, so prompts this miner renders
    differently from the validator would otherwise be generated again forever.
    """
    import heapq

    counts: Counter[str] = Counter()
    retry_kwargs = dict(sleep=sleep, counts=counts, max_consecutive_failures=max_consecutive_failures)
    n = job.sampling.n
    skipping = _can_skip(job, client, sign_skip)
    ledger = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
    next_cursor = ledger
    slots: dict[int, _Slot] = {}
    owner: dict[str, int] = {}
    redo: list[int] = []
    started = 0
    consecutive_unreasoned = 0
    consecutive_mismatches = 0
    complete = False

    def generating() -> int:
        return sum(1 for slot in slots.values() if len(slot.generations) < n)

    if backlog is not None:
        for cursor, (index, rendered, generations) in sorted(backlog.load(ledger).items()):
            if (len(generations) != n or index != job_walk_index(job, hotkey, cursor)
                    or rendered != render(index)):
                backlog.discard(cursor)
                continue
            request_ids = [f"stored-{cursor}-{i}" for i in range(n)]
            slots[cursor] = _Slot(index, rendered, prompt_token_ids(tokenizer, rendered),
                                  request_ids, dict(zip(request_ids, generations)))
        if slots:
            logger.info("resuming %d finished prompt(s) from the backlog, cursors %d..%d",
                        len(slots), min(slots), max(slots))

    room = getattr(generator, "room", None)
    prepared: dict[int, tuple[int, str, list[int]]] = {}

    def start(cursor: int) -> bool:
        """Start the prompt at ``cursor``; False when the KV cache has no room for
        it yet. Something must already be generating then: its finish frees room."""
        if cursor not in prepared:
            index = job_walk_index(job, hotkey, cursor)
            rendered = render(index)
            prepared[cursor] = (index, rendered, prompt_token_ids(tokenizer, rendered))
        index, rendered, prompt_ids = prepared[cursor]
        if room is not None and generating() and not room(len(prompt_ids), n):
            return False
        del prepared[cursor]
        slot = _Slot(index, rendered, prompt_ids, generator.start(prompt_ids, n), {})
        slots[cursor] = slot
        for request_id in slot.request_ids:
            owner[request_id] = cursor
        return True

    def drop(cursor: int) -> None:
        slot = slots.pop(cursor)
        generator.cancel([r for r in slot.request_ids if r not in slot.generations])
        for request_id in slot.request_ids:
            owner.pop(request_id, None)

    def resync() -> None:
        nonlocal ledger, next_cursor
        ledger = _retry(lambda: client.cursor(hotkey), **retry_kwargs)
        for cursor in [c for c in slots if c < ledger]:
            drop(cursor)
            if backlog is not None:
                backlog.discard(cursor)
        for cursor in [c for c in prepared if c < ledger]:
            del prepared[cursor]
        if ledger >= next_cursor:
            next_cursor = ledger
        elif ledger not in slots:
            # The ledger stayed on a refused prompt: generate it again.
            heapq.heappush(redo, ledger)

    def fill() -> None:
        # max_steps counts new cursors: a prompt generated again is not a new step.
        nonlocal ledger, next_cursor, started, skipping, complete
        while not complete and generating() < window:
            if redo:
                cursor = heapq.heappop(redo)
                if cursor < ledger or cursor in slots:
                    continue
                if not start(cursor):
                    heapq.heappush(redo, cursor)
                    return
                continue
            if len(slots) >= max_lookahead or (max_steps is not None and started >= max_steps):
                return
            else:
                if skipping and next_cursor == ledger and not slots and not submitting:
                    cursor, skipping = _past_full_prompts(
                        job=job, hotkey=hotkey, client=client, cursor=ledger,
                        sign_skip=sign_skip, counts=counts, retry_kwargs=retry_kwargs,
                    )
                    if cursor is None:
                        complete = True
                        return
                    ledger = next_cursor = cursor
                while next_cursor in slots:
                    next_cursor += 1
                cursor = next_cursor
            if not start(cursor):
                return
            next_cursor += 1
            started += 1

    submitting: list = []

    def submittable() -> bool:
        return (not complete and not submitting and ledger in slots
                and len(slots[ledger].generations) == n)

    def submit_head() -> None:
        """Send the ledger prompt; the answer is taken by ``answered``. One at a
        time: the route only takes the ledger cursor, which the answer moves."""
        cursor = ledger
        slot = slots.pop(cursor)
        for request_id in slot.request_ids:
            owner.pop(request_id, None)
        body = build_submission(
            job=job, hotkey=hotkey, cursor=cursor, prompt_index=slot.prompt_index,
            rendered_prompt=slot.rendered,
            generations=[slot.generations[r] for r in slot.request_ids],
            tokenizer=tokenizer, sign=sign,
        )
        submitting.append((cursor, slot, submitter.submit(
            _retry, lambda: client.submit(body), **retry_kwargs)))

    def answered() -> None:
        nonlocal ledger, consecutive_unreasoned, consecutive_mismatches, complete
        cursor, slot, future = submitting.pop()
        answer = future.result()
        if backlog is not None:
            backlog.discard(cursor)
        reason = answer.get("reason")
        if reason is None:
            consecutive_unreasoned += 1
            counts["permanent_failure"] += 1
            logger.error("corpus submission answered with no reason: %r", answer)
            if consecutive_unreasoned >= max_consecutive_failures:
                raise CorpusMinerHalted(
                    f"{consecutive_unreasoned} consecutive corpus answers carried no reason",
                    counts=dict(counts),
                )
            resync()
            return
        consecutive_unreasoned = 0
        reason = str(reason)
        counts[reason] += 1
        logger.info("cursor %d prompt %d: %s, %d tokens (%d generating)",
                    cursor, slot.prompt_index, reason,
                    sum(len(g.tokens) for g in slot.generations.values()), generating())
        if reason == "job_complete":
            complete = True
            return
        if reason in _HALT:
            raise CorpusMinerHalted(f"the validator refused this hotkey: {reason}",
                                    counts=dict(counts))
        consecutive_mismatches = consecutive_mismatches + 1 if reason == "prompt_mismatch" else 0
        if consecutive_mismatches >= max_prompt_mismatches:
            counts["stopped_prompt_mismatch"] += 1
            logger.error("corpus job %s: %d consecutive prompt_mismatch answers (%s); "
                         "stopping it", job.job_id, consecutive_mismatches, answer.get("detail"))
            complete = True
            return
        if answer.get("accepted"):
            ledger += 1
            return
        if reason not in _RESYNC:
            logger.warning("corpus submission refused: %s %s", reason, answer.get("detail"))
        resync()

    # The route answers in seconds to a minute; submitting on this thread would
    # leave the GPU idle for every one of them.
    own_submitter = submit_executor is None
    if own_submitter:
        from concurrent.futures import ThreadPoolExecutor

        submit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpus-submit")
    submitter = submit_executor
    heartbeat = time.monotonic()
    finished_tokens = 0
    try:
        while True:
            fill()
            if submitting and submitting[0][2].done():
                answered()
            if submittable():
                submit_head()
            if time.monotonic() - heartbeat >= 300:
                logger.info("%d generating, %d finished waiting on cursor %d, %.1f tokens/s finished",
                            generating(), len(slots) - generating(), ledger,
                            finished_tokens / (time.monotonic() - heartbeat))
                heartbeat, finished_tokens = time.monotonic(), 0
            if complete:
                break
            if not generator.busy():
                fill()
                if not generator.busy():
                    if submitting:
                        submitting[0][2].result()
                        continue
                    if submittable():
                        continue
                    # Nothing is generating or submitting and the head cannot
                    # be submitted: only max_steps can leave the window like this.
                    break
            for request_id, tokens in generator.step():
                cursor = owner.get(request_id)
                if cursor is None:
                    continue
                slot = slots[cursor]
                try:
                    slot.generations[request_id] = generator.finish(
                        request_id, len(slot.prompt_ids), tokens)
                    finished_tokens += len(tokens)
                    if backlog is not None and len(slot.generations) == n:
                        backlog.save(cursor, slot.prompt_index, slot.rendered,
                                     [slot.generations[r] for r in slot.request_ids])
                except ValueError as exc:
                    logger.warning("dropping a corpus generation at cursor %d: %s", cursor, exc)
                    counts["generation_failed"] += 1
                    drop(cursor)
                    heapq.heappush(redo, cursor)
    except CorpusJobRetired as exc:
        counts["job_retired"] += 1
        logger.info("corpus job %s is retired; stopping it (%s)", job.job_id, exc)
    finally:
        for cursor in list(slots):
            drop(cursor)
        if own_submitter:
            # A submission still in flight finishes on its own; its prompt stays
            # in the backlog until a restart reads the ledger past it.
            submitter.shutdown(wait=False, cancel_futures=True)
    return dict(counts)


class SharedEngine:
    """One generator serving several jobs' ``mine_window``, each on its own thread.

    vLLM and the hidden-state capture are touched only by this engine's pump
    thread: a view's ``start``, ``cancel``, ``room`` and row reads are queued to
    it and run between engine steps, and every finished request is handed to
    the view that started it. A job blocked on the validator (a cursor read, a
    503 backoff) therefore never stalls the engine for the others.
    """

    def __init__(self, generator, *, idle_wait: float = 0.2) -> None:
        import queue
        import threading

        self._generator = generator
        self._calls: queue.SimpleQueue = queue.SimpleQueue()
        self._owner: dict[str, _EngineView] = {}
        self._idle_wait = idle_wait
        self._stopped = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._pump, name="engine", daemon=True)
        self._thread.start()

    def view(self, max_tokens: int | None = None) -> "_EngineView":
        """A job's view; ``max_tokens`` its completion budget when it is not
        the generator's own."""
        return _EngineView(self, max_tokens)

    def call(self, fn, *args):
        """Run ``fn(*args)`` on the pump thread and return its result."""
        from concurrent.futures import Future

        if self._error is not None or self._stopped.is_set():
            raise RuntimeError("the shared engine stopped") from self._error
        future: Future = Future()
        self._calls.put((fn, args, future))
        return future.result()

    def _run_calls(self, block: bool) -> None:
        import queue

        try:
            item = self._calls.get(timeout=self._idle_wait) if block else self._calls.get_nowait()
        except queue.Empty:
            return
        while True:
            fn, args, future = item
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # handed to the calling job's thread
                future.set_exception(exc)
            try:
                item = self._calls.get_nowait()
            except queue.Empty:
                return

    def _pump(self) -> None:
        try:
            while not self._stopped.is_set():
                busy = self._generator.busy()
                self._run_calls(block=not busy)
                if not busy:
                    continue
                for request_id, tokens in self._generator.step():
                    view = self._owner.pop(request_id, None)
                    if view is not None:
                        view._deliver(request_id, tokens)
        except BaseException as exc:
            logger.exception("the shared engine stopped")
            self._error = exc
        finally:
            self._stopped.set()
            for view in set(self._owner.values()):
                view._fail()
            self._fail_waiting_calls()

    def _fail_waiting_calls(self) -> None:
        import queue

        while True:
            try:
                _, _, future = self._calls.get_nowait()
            except queue.Empty:
                return
            future.set_exception(RuntimeError("the shared engine stopped"))

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=30)
        self._fail_waiting_calls()


class _EngineView:
    """A ``WindowGenerator`` over a ``SharedEngine``, scoped to one job's requests."""

    def __init__(self, engine: SharedEngine, max_tokens: int | None = None) -> None:
        import queue

        self._engine = engine
        self._budget = {} if max_tokens is None else {"max_tokens": max_tokens}
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        # Changed only on the pump thread; read by the job's.
        self._pending = 0

    def _deliver(self, request_id: str, tokens: list[int]) -> None:
        self._inbox.put((request_id, tokens))
        self._pending -= 1

    def _fail(self) -> None:
        self._inbox.put(None)

    def _start(self, prompt_ids, n):
        request_ids = self._engine._generator.start(prompt_ids, n, **self._budget)
        for request_id in request_ids:
            self._engine._owner[request_id] = self
        self._pending += len(request_ids)
        return request_ids

    def start(self, prompt_ids: list[int], n: int) -> list[str]:
        return self._engine.call(self._start, prompt_ids, n)

    def _cancel(self, request_ids):
        generator = self._engine._generator
        owned = [r for r in request_ids if self._engine._owner.get(r) is self]
        for request_id in owned:
            del self._engine._owner[request_id]
        self._pending -= len(owned)
        generator.cancel(owned)
        # Finished but not yet taken from the inbox: nothing to abort, but their
        # rows are still captured.
        finished = [r for r in request_ids if r not in owned]
        if finished and callable(getattr(generator, "forget", None)):
            generator.forget(finished)

    def cancel(self, request_ids: list[str]) -> None:
        if request_ids:
            self._engine.call(self._cancel, list(request_ids))

    def _room(self, prompt_len, n):
        return self._engine._generator.room(prompt_len, n, **self._budget)

    def room(self, prompt_len: int, n: int) -> bool:
        return self._engine.call(self._room, prompt_len, n)

    def finish(self, request_id: str, prompt_len: int, tokens: list[int]) -> Generation:
        generator = self._engine._generator
        rows = self._engine.call(generator.rows_for, request_id, prompt_len, tokens)
        return generator.prove(rows, tokens)

    def step(self) -> list[tuple[str, list[int]]]:
        import queue

        try:
            items = [self._inbox.get(timeout=self._engine._idle_wait)]
        except queue.Empty:
            if self._engine._stopped.is_set():
                raise RuntimeError("the shared engine stopped") from self._engine._error
            return []
        while True:
            try:
                items.append(self._inbox.get_nowait())
            except queue.Empty:
                break
        if any(item is None for item in items):
            raise RuntimeError("the shared engine stopped") from self._engine._error
        return items

    def busy(self) -> bool:
        return self._pending > 0 or not self._inbox.empty()

    def window(self, n: int) -> int:
        return self._engine._generator.window(n)


@dataclass
class JobRun:
    """One job of ``mine_jobs``: ``mine_window``'s keyword arguments but the
    generator, and the job's completion budget when it is shorter than the
    generator's."""

    name: str
    kwargs: dict
    max_tokens: int | None = None


def mine_jobs(runs: list[JobRun], generator) -> dict[str, dict]:
    """Mine several jobs at once on one generator, each with ``mine_window`` on
    its own thread (named after the job). One job ending -- complete, retired,
    stopped -- leaves the others mining; a hotkey refusal or an unexpected error
    in any job stops them all and is raised once every thread has returned."""
    import threading

    engine = SharedEngine(generator)
    results: dict[str, dict] = {}
    failures: list[tuple[str, BaseException]] = []
    stop = threading.Event()

    def run(job_run: JobRun) -> None:
        try:
            view = engine.view(job_run.max_tokens)
            results[job_run.name] = mine_window(generator=view, **job_run.kwargs)
        except BaseException as exc:
            failures.append((job_run.name, exc))
            stop.set()

    threads = [threading.Thread(target=run, args=(job_run,), name=job_run.name, daemon=True)
               for job_run in runs]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            if stop.is_set():
                # The other jobs' next engine call raises, which ends them.
                engine.stop()
            for thread in threads:
                thread.join(timeout=1.0)
    finally:
        engine.stop()
    if failures:
        name, exc = failures[0]
        logger.error("corpus job %s stopped every job: %s", name, exc)
        raise exc
    return results


# Room for the rendered prompt beside the job's completion budget.
PROMPT_ALLOWANCE_TOKENS = 8192
# A step runs n sequences; vLLM's default 1,024 overruns a hybrid model's Mamba cache.
MAX_NUM_SEQS = 256
# Admitting by free KV blocks: the most prompts generating at once unless the
# operator caps it, and the growth each new prompt must find room for beyond
# its prompt (most completions end well inside it; longer ones take the spare
# share, or are preempted and recomputed).
OVERCOMMIT_MAX_IN_FLIGHT = 32
OVERCOMMIT_GROWTH_TOKENS = 4096


def _has_vision_encoder(checkpoint_dir: str) -> bool:
    import json
    from pathlib import Path

    try:
        config = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and "vision_config" in config


class CorpusContractError(RuntimeError):
    """The validator served a contract that does not describe this job."""


def save_served_contract(contract: dict, job, directory) -> "Path":
    """Keep the contract the validator serves, once it is known to describe the
    job's checkpoint and to carry the toploc proof every submission needs."""
    import json
    from pathlib import Path

    if (contract.get("model_id") != job.checkpoint_repo
            or contract.get("model_revision") != job.checkpoint_revision):
        raise CorpusContractError(
            f"the served contract describes {contract.get('model_id')!r}@"
            f"{contract.get('model_revision')!r}, not the job's "
            f"{job.checkpoint_repo!r}@{job.checkpoint_revision!r}"
        )
    if not any(p.get("scheme") == "toploc-v1" for p in contract.get("proofs") or ()):
        raise CorpusContractError("the served contract carries no toploc proof")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job.job_id}.contract.json"
    path.write_text(json.dumps(contract, sort_keys=True, separators=(",", ":")))
    return path


class VllmGenerator:
    """vLLM in-process on the V1 runner, capturing decode activations for proofs.

    One request per completion (n=1 each), so every captured row set maps to
    exactly one completion; prefix caching is off because cached rows are never
    recomputed and would be missing from the capture.
    """

    def __init__(self, checkpoint_dir: str, sampling, proof, eos_token_id: int,
                 gpu_memory_utilization: float | None = None, *,
                 speculative_tokens: int = 0, kv_headroom: float | None = None) -> None:
        """``speculative_tokens``: draft tokens per step from the checkpoint's
        MTP head (0 = off). ``kv_headroom``: admit prompts by the KV blocks
        actually free, keeping this share of the pool spare, rather than
        reserving every prompt's full length; a request vLLM then has to
        preempt is recomputed. Both need the capture to place rows by position."""
        import os

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        from vllm import LLM, SamplingParams

        from reliquary.miner.vllm_hidden_capture import capture_hidden_states

        self._headroom = kv_headroom
        self._capture_cm = capture_hidden_states(
            non_blocking=True, positional=bool(speculative_tokens) or kv_headroom is not None)
        self._capture = self._capture_cm.__enter__()
        # Only passed when set: a miner alone on its card keeps vLLM's own
        # default, one sharing it (e.g. with a validator) asks for less.
        memory = {} if gpu_memory_utilization is None else {"gpu_memory_utilization": gpu_memory_utilization}
        # vLLM seeds every engine with 0: two miners sampling one prompt at the
        # same step would submit identical completions, the second refused
        # hash_duplicate. The audit never depends on the seed.
        seed = secrets.randbelow(2**31 - 1) + 1
        # vLLM otherwise reserves the checkpoint's own maximum length, whose KV
        # cache need not fit the card; the job never asks for more than this.
        extra = {"max_model_len": sampling.max_new_tokens + PROMPT_ALLOWANCE_TOKENS,
                 "max_num_seqs": MAX_NUM_SEQS}
        if _has_vision_encoder(checkpoint_dir):
            # The job's prompts are text: skip the vision encoder's profiling.
            extra["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        if speculative_tokens:
            extra["speculative_config"] = {"method": "mtp", "num_speculative_tokens": speculative_tokens}
        self._llm = LLM(model=checkpoint_dir, dtype="bfloat16", enable_prefix_caching=False,
                        seed=seed, **memory, **extra)
        self._sampling_kwargs = dict(
            n=1, temperature=sampling.temperature, top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k > 0 else -1,
            min_tokens=sampling.min_new_tokens, max_tokens=sampling.max_new_tokens,
            # The job's eos is the only terminator `check_text_matches_tokens`
            # judges against; the checkpoint's own generation_config may list
            # others (e.g. both im_end and endoftext), and vLLM stopping on
            # one of those instead would get an honest completion refused
            # bad_termination. `ignore_eos=True` turns off that model-config
            # default so only `stop_token_ids` below can end generation --
            # `min_tokens` above still masks it until the floor is reached.
            # Checked on vLLM 0.30 (H100, 2026-09-24): the stop token is kept
            # at the end of `token_ids`, which `completion_text` relies on.
            stop_token_ids=[eos_token_id], ignore_eos=True,
        )
        self._params = SamplingParams(**self._sampling_kwargs)
        # Shorter budgets for other jobs on this engine; never longer, since
        # max_model_len was sized for this one.
        self._params_by_budget = {sampling.max_new_tokens: self._params}
        self._proof = proof
        self._engine_ids: dict[str, str] = {}
        self._reserved: dict[str, int] = {}
        try:
            self._kv = self._kv_layout()
        except Exception as exc:  # vLLM internals moved: size by the reported concurrency
            logger.warning("cannot read vLLM's KV cache layout (%s): sizing for full length", exc)
            self._kv = None

    def _kv_layout(self) -> tuple[int, list[int], int, int] | None:
        """(blocks in the pool, block size of each full-attention group, blocks a
        request holds whatever its length, max_model_len), checked against the
        concurrency vLLM reports so a misread layout is never trusted."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        engine = self._llm.llm_engine
        vllm_config = engine.vllm_config
        kv_config = engine.engine_core.engine_core.scheduler.kv_cache_config
        block_sizes, fixed = [], 0
        for group in kv_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, FullAttentionSpec) and spec.sliding_window is None:
                block_sizes.append(spec.block_size)
            else:
                # Bounded by its use at max_model_len, so never under-counted.
                fixed += -(-spec.max_memory_usage_bytes(vllm_config) // spec.page_size_bytes)
        max_len = vllm_config.model_config.max_model_len
        layout = (kv_config.num_blocks, block_sizes, fixed, max_len)
        reported = getattr(vllm_config.cache_config, "kv_cache_max_concurrency", None)
        derived = kv_config.num_blocks / self._blocks_for(layout, max_len)
        if not reported or abs(derived - reported) > 0.01 * reported:
            logger.warning("KV layout gives %.2f requests at full length, vLLM reports %s: "
                           "sizing for full length", derived, reported)
            return None
        return layout

    @staticmethod
    def _blocks_for(layout, length: int) -> int:
        _, block_sizes, fixed, max_len = layout
        length = min(length, max_len)
        return fixed + sum(-(-length // size) for size in block_sizes)

    def _params_for(self, max_tokens: int | None):
        if max_tokens is None:
            return self._params
        params = self._params_by_budget.get(max_tokens)
        if params is None:
            from vllm import SamplingParams

            if max_tokens > self._params.max_tokens:
                raise ValueError(f"a {max_tokens}-token budget exceeds this engine's "
                                 f"{self._params.max_tokens}")
            params = self._params_by_budget[max_tokens] = SamplingParams(
                **{**self._sampling_kwargs, "max_tokens": max_tokens})
        return params

    def _request_blocks(self, prompt_len: int, max_tokens: int | None = None) -> int:
        # +1: async scheduling may run one step past the last token.
        budget = self._params.max_tokens if max_tokens is None else max_tokens
        return self._blocks_for(self._kv, prompt_len + budget + 1)

    def _scheduler(self):
        return self._llm.llm_engine.engine_core.engine_core.scheduler

    def room(self, prompt_len: int, n: int, max_tokens: int | None = None) -> bool:
        """Whether ``n`` more requests for this prompt fit the KV cache beside those
        generating, every one at its own full length (prompt plus max_tokens);
        with a headroom, whether the blocks actually free cover the prompt and
        its first growth with the spare share left over."""
        if self._headroom is not None:
            scheduler = self._scheduler()
            # A queued request holds no blocks yet, so the free count below
            # does not know about it: one admission per step, and none while a
            # preempted request waits to be recomputed.
            if len(scheduler.waiting):
                return False
            if not self._reserved:
                return True
            pool = scheduler.kv_cache_manager.block_pool
            need = 0 if self._kv is None else n * self._blocks_for(self._kv, prompt_len + OVERCOMMIT_GROWTH_TOKENS)
            return pool.get_num_free_blocks() - need >= self._headroom * pool.num_gpu_blocks
        if self._kv is None or not self._reserved:
            return True
        # The pool's first block is vLLM's null block, never handed out.
        free = self._kv[0] - 1 - sum(self._reserved.values())
        return n * self._request_blocks(prompt_len, max_tokens) <= free

    def capacity(self, prompt_len: int, n: int) -> int:
        """How many prompts of ``prompt_len`` tokens ``room`` lets generate at once."""
        if self._kv is None or self._headroom is not None:
            return self.window(n)
        return max(1, (self._kv[0] - 1) // (n * self._request_blocks(prompt_len)))

    def generate(self, prompt_ids: list[int], n: int) -> list[Generation]:
        from vllm.inputs import TokensPrompt

        outputs = self._llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)] * n, self._params)
        return self._generations(outputs, [len(prompt_ids)] * n)

    def generate_many(self, prompts: list[list[int]], ns: list[int]) -> list[list[Generation]]:
        """Several prompts, ``ns[i]`` completions each, decoded in one batch
        (qualification); grouped back per prompt."""
        from vllm.inputs import TokensPrompt

        flat = [ids for ids, n in zip(prompts, ns) for _ in range(n)]
        outputs = self._llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in flat],
                                     self._params)
        generations = self._generations(outputs, [len(ids) for ids in flat])
        grouped, start = [], 0
        for n in ns:
            grouped.append(generations[start:start + n])
            start += n
        return grouped

    def _generations(self, outputs, prompt_lengths: list[int]) -> list[Generation]:
        import base64

        from reliquary.miner.vllm_hidden_capture import completion_rows
        from reliquary.protocol.toploc_proof import build_chunk_proofs

        generations = []
        for index, (output, prompt_length) in enumerate(zip(outputs, prompt_lengths)):
            tokens = list(output.outputs[0].token_ids)
            # `pop`, not `for_request`: this generator lives for the whole
            # mining run, and a request's rows are never read again after its
            # proof is built, so keeping them would grow CPU memory unbounded.
            try:
                total = prompt_length + len(tokens)
                rows = completion_rows(self._capture.pop(output.request_id, limit=total - 1),
                                       prompt_length, total)
            except ValueError:
                # A row-count mismatch usually means vLLM preempted and
                # recomputed this request under KV pressure mid-batch: the
                # rest of this batch's captured rows are just as suspect, and
                # every one still in `self._capture` would otherwise sit
                # there forever. Forget them, then let `mine_steps` drop the
                # whole step rather than submit generations no proof can be
                # trusted for.
                for leftover in outputs[index + 1:]:
                    try:
                        self._capture.pop(leftover.request_id)
                    except KeyError:
                        pass
                raise
            proofs = build_chunk_proofs(rows, chunk_tokens=self._proof.chunk_tokens, topk=self._proof.topk)
            generations.append(Generation(tokens, [base64.b64encode(p).decode() for p in proofs]))
        return generations

    def window(self, n: int) -> int:
        """The most prompts of ``n`` requests to generate at once. With the KV
        layout known, ``room`` admits each prompt by its own length and this is
        only the scheduler's sequence cap; otherwise, as many as fit with every
        request at the full ``max_model_len``. Either way the scheduler never has
        to preempt. With a headroom, ``room`` admits by the blocks free and this
        caps the prompts generating at once."""
        if self._headroom is not None:
            return max(1, OVERCOMMIT_MAX_IN_FLIGHT // n)
        if self._kv is not None:
            return max(1, MAX_NUM_SEQS // n)
        cache = self._llm.llm_engine.vllm_config.cache_config
        concurrency = getattr(cache, "kv_cache_max_concurrency", None)
        if not concurrency:
            logger.warning("vLLM did not report its KV cache concurrency: one prompt at a time")
            return 1
        return max(1, min(int(concurrency), MAX_NUM_SEQS) // n)

    def start(self, prompt_ids: list[int], n: int, max_tokens: int | None = None) -> list[str]:
        """Queue ``n`` requests for one prompt, each completing in at most
        ``max_tokens`` (the engine's job's budget by default); their engine
        request ids."""
        from vllm.inputs import TokensPrompt

        request_ids = list(self._llm.enqueue([TokensPrompt(prompt_token_ids=prompt_ids)] * n,
                                             self._params_for(max_tokens), use_tqdm=False))
        blocks = self._request_blocks(len(prompt_ids), max_tokens) if self._kv is not None else 0
        for request_id in request_ids:
            # Outputs name the id before the engine's "-<random>" suffix.
            self._engine_ids[request_id.split("-", 1)[0]] = request_id
            self._reserved[request_id] = blocks
        return request_ids

    def step(self) -> list[tuple[str, list[int]]]:
        """One engine step; the requests it finished, by ``start``'s id, with their tokens."""
        finished = []
        for output in self._llm.llm_engine.step():
            if output.finished:
                request_id = self._engine_ids.pop(output.request_id, output.request_id)
                self._reserved.pop(request_id, None)
                finished.append((request_id, list(output.outputs[0].token_ids)))
        return finished

    def finish(self, request_id: str, prompt_len: int, tokens: list[int]) -> Generation:
        """Prove a finished request; ValueError when its captured rows do not fit it."""
        return self.prove(self.rows_for(request_id, prompt_len, tokens), tokens)

    def rows_for(self, request_id: str, prompt_len: int, tokens: list[int]):
        """A finished request's completion rows, taken out of the capture."""
        from reliquary.miner.vllm_hidden_capture import completion_rows

        total = prompt_len + len(tokens)
        return completion_rows(self._capture.pop(request_id, limit=total - 1), prompt_len, total)

    def prove(self, rows, tokens: list[int]) -> Generation:
        import base64

        from reliquary.protocol.toploc_proof import build_chunk_proofs

        proofs = build_chunk_proofs(rows, chunk_tokens=self._proof.chunk_tokens, topk=self._proof.topk)
        return Generation(tokens, [base64.b64encode(p).decode() for p in proofs])

    def cancel(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        self._llm.llm_engine.abort_request(list(request_ids), internal=True)
        for request_id in request_ids:
            self._engine_ids.pop(request_id.split("-", 1)[0], None)
            self._reserved.pop(request_id, None)
        self.forget(request_ids)

    def forget(self, request_ids: list[str]) -> None:
        """Drop the captured rows of requests that will never be proved."""
        for request_id in request_ids:
            try:
                self._capture.pop(request_id)
            except KeyError:
                pass

    def busy(self) -> bool:
        return self._llm.llm_engine.has_unfinished_requests()
