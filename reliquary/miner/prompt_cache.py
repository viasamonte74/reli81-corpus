"""A job's prompt rows, rendered once by a separate interpreter.

Some prompt sources ship as packages whose pins conflict with the miner's
vLLM environment. Their rows are written here by a Python that has the
package (``python -m reliquary.miner.prompt_cache``, under the job's served
contract so the environment renders through that contract's profile) and read
by the miner, which wraps each row in the checkpoint's chat template as the
validator does. The header binds the file to one job's rows, so a cache for
another job, range or environment build is refused rather than mined.

No import here may need torch, vLLM or the prompt source's package.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "reliquary/corpus-prompt-cache/v1"


class PromptCacheError(ValueError):
    pass


def _header(job: dict, *, profile_id: str, manifest_sha256: str | None) -> dict:
    return {
        "schema": SCHEMA,
        "job_id": job["job_id"],
        "prompt_source": job["prompt_source"],
        "prompt_start": int(job.get("prompt_start") or 0),
        "prompt_count": int(job["prompt_count"]),
        "profile_id": profile_id,
        "environment_manifest_sha256": manifest_sha256,
    }


def write_prompt_cache(job: dict, path) -> dict:
    """Render every row the job owns into ``path``; the header written."""
    from reliquary.environment.registry import get_environment_spec
    from reliquary.protocol import profiles

    profile = profiles.ACTIVE_PROTOCOL_PROFILE
    if job["prompt_source"] not in profile.environments:
        raise PromptCacheError(
            f"the active profile {profile.profile_id!r} declares no environment "
            f"{job['prompt_source']!r}: run under the job's served contract")
    spec = get_environment_spec(job["prompt_source"])
    if getattr(spec, "interaction_mode", None) != "single_turn":
        raise PromptCacheError(f"{job['prompt_source']!r} is not a single-turn source")
    header = _header(job, profile_id=profile.profile_id,
                     manifest_sha256=getattr(spec, "environment_manifest_sha256", None))
    environment = spec.create()
    start, count = header["prompt_start"], header["prompt_count"]
    if len(environment) < start + count:
        raise PromptCacheError(
            f"{job['prompt_source']!r} has {len(environment)} rows, the job ends at {start + count}")
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as out:
        out.write(json.dumps(header, sort_keys=True) + "\n")
        for position in range(start, start + count):
            problem = environment.get_problem(position)
            prompt = problem.get("prompt") if isinstance(problem, dict) else None
            if not isinstance(prompt, str) or not prompt:
                raise PromptCacheError(f"row {position} has no prompt text")
            out.write(json.dumps(prompt) + "\n")
    temp.replace(path)
    return header


class PromptCache:
    """The rows of one job, by source index."""

    def __init__(self, path, job) -> None:
        with Path(path).open() as source:
            try:
                header = json.loads(source.readline())
            except ValueError as exc:
                raise PromptCacheError(f"{path}: unreadable header") from exc
            expected = {"schema": SCHEMA, "job_id": job.job_id,
                        "prompt_source": job.prompt_source,
                        "prompt_start": int(job.prompt_start),
                        "prompt_count": int(job.prompt_count)}
            for key, value in expected.items():
                if header.get(key) != value:
                    raise PromptCacheError(
                        f"{path}: {key} is {header.get(key)!r}, the job's is {value!r}")
            self._rows = [json.loads(line) for line in source]
        if len(self._rows) != job.prompt_count:
            raise PromptCacheError(
                f"{path}: {len(self._rows)} rows, the job has {job.prompt_count}")
        self.header = header
        self._start = int(job.prompt_start)

    def prompt(self, prompt_index: int) -> str:
        offset = int(prompt_index) - self._start
        if not 0 <= offset < len(self._rows):
            raise PromptCacheError(f"prompt {prompt_index} is outside the cached rows")
        return self._rows[offset]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="the job manifest as served (JSON)")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_prompt_cache(json.loads(args.job.read_text()), args.out), indent=2))
