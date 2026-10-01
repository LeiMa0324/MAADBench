"""Compatibility entry point; uses the shared action-level Gemma runner."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark_AD.methods.llm_as_judge.run_gemma_4 import main
if __name__ == "__main__":
    main()
