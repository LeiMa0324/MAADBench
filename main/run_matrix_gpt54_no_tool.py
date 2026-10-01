"""Run the no-tool room/config matrix for GPT-5.4."""

from main.run_trace_matrix import main


if __name__ == "__main__":
    raise SystemExit(main(("gpt_54",), unit_tools="disabled"))
