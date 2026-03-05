from sage_generator import *


gen = SAGEDataGenerator(seed=42)
tasks = gen.generate(easy=30, medium=30, hard=40)
save_jsonl(tasks, "tasks.jsonl")

gen = SAGEDataGenerator(seed=42)
tasks = gen.generate(
    easy=30, medium=30, hard=40,
    anomaly_modes={"FM-2.6": 5, "FM-3.1": 3},
    anomaly_levels=["easy"],
)
save_jsonl(tasks, "injected_tasks.jsonl")