"""Run EscapeRoom with GSM-hard clues."""


def main():
    from main.runner import EscapeRoomRunner, parse_args
    args = parse_args()
    args.clue_domain = "gsm-hard"
    EscapeRoomRunner(args).run()


if __name__ == "__main__":
    main()
