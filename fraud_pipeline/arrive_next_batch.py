"""A new batch of labeled data 'arrives': move it from pending/ to landing/. Quiet no-op if nothing is waiting."""
from lakehouse import arrive_next_batch

batch = arrive_next_batch()
print(f"arrived: {batch}" if batch else "no new batch")
