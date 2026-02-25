"""
SWEDataLoader — loads swe_bench_verified.jsonl into SWEInstance objects.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple


# ── Verify result ─────────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    resolved: bool
    fail_to_pass_passed: List[str] = dc_field(default_factory=list)
    fail_to_pass_failed: List[str] = dc_field(default_factory=list)
    pass_to_pass_passed: List[str] = dc_field(default_factory=list)
    pass_to_pass_failed: List[str] = dc_field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "fail_to_pass_passed": self.fail_to_pass_passed,
            "fail_to_pass_failed": self.fail_to_pass_failed,
            "pass_to_pass_passed": self.pass_to_pass_passed,
            "pass_to_pass_failed": self.pass_to_pass_failed,
            "error": self.error,
        }


def _run_tests(
    work_dir: str,
    test_ids: List[str],
    timeout: int = 120,
) -> Tuple[List[str], List[str], bool]:
    """
    Run pytest on test_ids inside work_dir.
    Returns (passed, failed, all_passed).

    all_passed is determined solely by pytest's exit code (0 = all passed),
    which is more reliable than text parsing.  passed/failed lists are
    best-effort detail parsed from --junit-xml output.
    """
    if not test_ids:
        return [], [], True  # vacuously true: nothing to fail

    xml_file = os.path.join(tempfile.gettempdir(), f"swe_junit_{os.getpid()}.xml")
    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=no", "-q",
        f"--junit-xml={xml_file}",
    ] + test_ids

    try:
        result = subprocess.run(
            cmd, cwd=work_dir,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return [], list(test_ids), False

    # ── Primary signal: exit code ─────────────────────────────────────────
    # pytest exits 0 iff every collected test passed (no failures/errors).
    all_passed = (result.returncode == 0)

    # ── Best-effort detail: parse JUnit XML ───────────────────────────────
    passed, failed = [], []
    if os.path.exists(xml_file):
        try:
            tree = ET.parse(xml_file)
            for tc in tree.iter("testcase"):
                classname = tc.get("classname", "")
                name      = tc.get("name", "")
                node_id   = f"{classname.replace('.', '/')}::{name}" if classname else name
                if tc.find("failure") is not None or tc.find("error") is not None:
                    failed.append(node_id)
                elif tc.find("skipped") is None:
                    passed.append(node_id)
        except ET.ParseError:
            pass
        finally:
            os.unlink(xml_file)

    # Fallback: no XML parsed — synthesize from exit code
    if not passed and not failed:
        if all_passed:
            passed = list(test_ids)
        else:
            failed = list(test_ids)

    return passed, failed, all_passed


# ── Instance ──────────────────────────────────────────────────────────────────

@dataclass
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    patch: str          # ground-truth patch (unified diff)
    test_patch: str
    fail_to_pass: list
    pass_to_pass: list
    difficulty: Optional[str] = None
    version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "difficulty": self.difficulty,
            "version": self.version,
        }

    def verify(
        self,
        generated_patch: str,
        repo_cache_dir: Optional[str] = None,
        test_timeout: int = 120,
    ) -> VerifyResult:
        """
        Evaluate generated_patch against this instance:
          1. Clone repo at base_commit (cached if repo_cache_dir given)
          2. Apply test_patch (introduces the failing tests)
          3. Apply generated_patch
          4. Run fail_to_pass tests (must pass) and pass_to_pass tests (must not regress)

        Args:
            generated_patch: unified diff output from the MAS
            repo_cache_dir:  directory for bare-clone cache (e.g. "~/.swe_cache").
                             Skips re-cloning across multiple verify() calls.
            test_timeout:    seconds allowed per test-suite run
        """
        def _empty_result(error: str) -> VerifyResult:
            return VerifyResult(
                resolved=False,
                fail_to_pass_failed=list(self.fail_to_pass),
                pass_to_pass_failed=list(self.pass_to_pass),
                error=error,
            )

        if not generated_patch or not generated_patch.strip():
            return _empty_result("empty patch")

        repo_url   = f"https://github.com/{self.repo}.git"
        work_dir   = None
        cached_dir = None

        try:
            # ── Set up working tree ───────────────────────────────────────────
            if repo_cache_dir:
                repo_cache_dir = os.path.expanduser(repo_cache_dir)
                os.makedirs(repo_cache_dir, exist_ok=True)
                cached_dir = os.path.join(repo_cache_dir, self.repo.replace("/", "__"))

                if not os.path.exists(cached_dir):
                    subprocess.run(
                        ["git", "clone", "--bare", repo_url, cached_dir],
                        check=True, capture_output=True, timeout=300,
                    )
                else:
                    subprocess.run(
                        ["git", "-C", cached_dir, "fetch", "--quiet", "origin"],
                        capture_output=True, timeout=60,
                    )

                work_dir = tempfile.mkdtemp(prefix=f"swe_{self.instance_id[:24]}_")
                subprocess.run(
                    ["git", "-C", cached_dir, "worktree", "add",
                     "--detach", work_dir, self.base_commit],
                    check=True, capture_output=True, timeout=60,
                )
            else:
                work_dir = tempfile.mkdtemp(prefix=f"swe_{self.instance_id[:24]}_")
                subprocess.run(
                    ["git", "clone", repo_url, work_dir],
                    check=True, capture_output=True, timeout=300,
                )
                subprocess.run(
                    ["git", "-C", work_dir, "checkout", self.base_commit],
                    check=True, capture_output=True, timeout=30,
                )

            # ── Install package ───────────────────────────────────────────────
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
                cwd=work_dir, capture_output=True, timeout=180,
            )

            # ── Apply test_patch ──────────────────────────────────────────────
            if self.test_patch and self.test_patch.strip():
                r = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    input=self.test_patch, text=True,
                    capture_output=True, cwd=work_dir,
                )
                if r.returncode != 0:
                    return _empty_result(f"test_patch apply failed: {r.stderr[:400]}")

            # ── Apply generated patch ─────────────────────────────────────────
            r = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                input=generated_patch, text=True,
                capture_output=True, cwd=work_dir,
            )
            if r.returncode != 0:
                return _empty_result(f"patch apply failed: {r.stderr[:400]}")

            # ── Run tests ─────────────────────────────────────────────────────
            f2p_pass, f2p_fail, f2p_ok = _run_tests(work_dir, self.fail_to_pass, timeout=test_timeout)
            p2p_pass, p2p_fail, p2p_ok = _run_tests(work_dir, self.pass_to_pass, timeout=test_timeout)

            # resolved = all fail_to_pass tests now pass  AND  no pass_to_pass regression.
            # f2p_ok / p2p_ok come from pytest exit code (0 = all passed),
            # which is the ground truth — not the text-parsed lists.
            return VerifyResult(
                resolved=f2p_ok and p2p_ok,
                fail_to_pass_passed=f2p_pass,
                fail_to_pass_failed=f2p_fail,
                pass_to_pass_passed=p2p_pass,
                pass_to_pass_failed=p2p_fail,
            )

        except subprocess.CalledProcessError as e:
            stderr = e.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return _empty_result(f"{e.cmd[0]}: {stderr[:400]}")
        except Exception as e:
            return _empty_result(str(e))
        finally:
            if work_dir and os.path.exists(work_dir):
                if cached_dir and os.path.exists(cached_dir):
                    subprocess.run(
                        ["git", "-C", cached_dir, "worktree", "remove", "--force", work_dir],
                        capture_output=True,
                    )
                else:
                    shutil.rmtree(work_dir, ignore_errors=True)


# ── Loader ────────────────────────────────────────────────────────────────────

class SWEDataLoader:
    """Loads and filters SWE-bench Verified instances from a JSONL file."""

    # Maps short keys to the raw difficulty strings in the dataset
    DIFFICULTY_MAP = {
        "easy":   "<15 min fix",
        "medium": "15 min - 1 hour",
        "hard":   "1-4 hours",
        "expert": ">4 hours",
    }

    def __init__(self, instances: List[SWEInstance], seed: int = 42):
        self._instances = instances
        self._seed = seed

    def __len__(self) -> int:
        return len(self._instances)

    def __iter__(self):
        return iter(self._instances)

    @classmethod
    def from_jsonl(cls, path: str, seed: int = 42) -> "SWEDataLoader":
        instances = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                # FAIL_TO_PASS / PASS_TO_PASS are JSON-encoded strings in the file
                fail_to_pass = data.get("FAIL_TO_PASS", "[]")
                pass_to_pass = data.get("PASS_TO_PASS", "[]")
                if isinstance(fail_to_pass, str):
                    fail_to_pass = json.loads(fail_to_pass)
                if isinstance(pass_to_pass, str):
                    pass_to_pass = json.loads(pass_to_pass)

                instances.append(SWEInstance(
                    instance_id=data["instance_id"],
                    repo=data["repo"],
                    base_commit=data["base_commit"],
                    problem_statement=data.get("problem_statement", ""),
                    hints_text=data.get("hints_text", ""),
                    patch=data.get("patch", ""),
                    test_patch=data.get("test_patch", ""),
                    fail_to_pass=fail_to_pass,
                    pass_to_pass=pass_to_pass,
                    difficulty=data.get("difficulty"),
                    version=data.get("version"),
                ))

        return cls(instances, seed=seed)

    def by_difficulty(self, difficulty: str, sample_size: int = -1) -> List[SWEInstance]:
        """Filter by short key ('easy','medium','hard','expert') or raw difficulty string.
        If sample_size > 0, randomly sample that many from the filtered list.
        """
        raw = self.DIFFICULTY_MAP.get(difficulty, difficulty)
        filtered = [i for i in self._instances if i.difficulty == raw]
        if sample_size > 0 and len(filtered) > sample_size:
            random.seed(self._seed)
            filtered = random.sample(filtered, sample_size)
        return filtered