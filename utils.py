from datetime import datetime

def generate_trace_id(dataset, mas_arch, query_id, label):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    trace_id  = f"{time}_{dataset}_{mas_arch}_{query_id}_{label}"
    return trace_id