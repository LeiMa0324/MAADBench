"""Run the no-tool room/config matrix for DeepSeek."""

from main.run_trace_matrix import main


TARGET_JOBS = {
    ("gsm-hard", "0.0"),
    ("gsm-hard", "0.3"),
    ("gsm-hard", "0.6"),
    ("livecodebench", "0.0"),
    ("livecodebench", "0.3"),
}


if __name__ == "__main__":
    raise SystemExit(main(("deepseek",), unit_tools="disabled", target_jobs=TARGET_JOBS))
