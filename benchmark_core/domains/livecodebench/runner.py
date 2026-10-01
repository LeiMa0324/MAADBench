"""Run EscapeRoom with LiveCodeBench execution clues."""

from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"


def main():
    from main.runner import EscapeRoomRunner, parse_args
    args = parse_args()
    args.clue_domain = "livecodebench"
    if args.output_dir == "output":
        args.output_dir = str(DEFAULT_OUTPUT_DIR)
    runner = EscapeRoomRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
