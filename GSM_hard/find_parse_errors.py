import json
import os
import glob

TRACES_DIR = os.path.join(os.path.dirname(__file__), "Turing_traces")

def main():
    files = sorted(glob.glob(os.path.join(TRACES_DIR, "*.json")))
    failed = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if "Failed to parse" in content:
            failed.append(os.path.basename(path))

    print(f"Total traces: {len(files)}, Failed to parse: {len(failed)}\n")
    for name in failed:
        print(name)


if __name__ == "__main__":
    main()