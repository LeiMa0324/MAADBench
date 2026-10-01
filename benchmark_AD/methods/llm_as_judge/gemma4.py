"""Export the shared split for inference in the LLM environment."""
import json
from pathlib import Path


def run_gemma4(args, run_setting, train_df=None, test_df=None):
    data_dir = Path(args.run_dir) / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [('train', train_df), ('test', test_df)]:
        frame.drop(columns=['raw_trace', 'step'], errors='ignore').to_csv(data_dir / f'{name}.csv', index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith('_')}
    config.pop('device', None)
    (data_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'LLM input exported to {data_dir}. Run python -m benchmark_AD.methods.llm_as_judge.run_gemma_4 --data-dir "{data_dir}" in the LLM environment.')
