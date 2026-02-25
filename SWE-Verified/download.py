from datasets import load_dataset

dataset = load_dataset('SWE-bench/SWE-bench_Verified', split='test')
print(len(dataset))  # 500 条

# 保存到本地
dataset.to_json('swe_bench_verified.jsonl')