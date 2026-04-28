import time
import psutil
import os
import pandas as pd
from app.engine import ImageRetrievalEngine


def run_benchmark(model_func, query, runs=5):
    process = psutil.Process(os.getpid())
    results = []

    for _ in range(runs):
        start_mem = process.memory_info().rss / (1024 * 1024)
        start_time = time.time()

        model_func(query)

        duration = time.time() - start_time
        cpu_load = psutil.cpu_percent(interval=None)
        mem_usage = (process.memory_info().rss / (1024 * 1024)) - start_mem

        results.append({"duration": duration, "cpu": cpu_load, "mem": max(0, mem_usage)})

    return pd.DataFrame(results).mean().to_dict()