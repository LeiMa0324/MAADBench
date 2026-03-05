import json

with open("traces/trace.json", "r") as f:
    data = json.load(f)


for agent_trace, step in zip(data["trace"], data["evaluation_result"]["evaluation"]["steps"]):
    if step['status'] !='correct':

        print("===============================")
        agent = agent_trace['agent']
        print(f"agent: {agent}, step: {step['step']}")
        if agent !='equivalence_checker':
            print(f"expression: {agent_trace['expression']}")
        else:
            print(f"expression: {agent_trace['final_expression']}")
        print(f"content: {agent_trace['output']['content']}")
        print("------ Evaluation -------")
        print(f'failure_mode: {step['failure_mode']}')
        print(f'detail: {step['detail']}')


