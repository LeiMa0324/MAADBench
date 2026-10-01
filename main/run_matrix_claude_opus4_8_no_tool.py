"""Run the no-tool room/config matrix for Claude Opus 4.8."""

from main.run_trace_matrix import main


if __name__ == "__main__":
    raise SystemExit(main(("claude_opus4.8",), unit_tools="disabled"))
