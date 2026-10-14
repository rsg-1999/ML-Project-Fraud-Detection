"""Split the labeled source files into landing/ (initial) and pending/ (3 arrival batches); copy the unlabeled test files."""
import os

import settings as S
from lakehouse import simulate_arrivals

missing = [f for f in ("client_train", "invoice_train", "client_test", "invoice_test")
           if not os.path.exists(f"{S.ROOT}/source/{f}.csv")]
assert not missing, f"upload these to {S.ROOT}/source/ first: {missing}"
simulate_arrivals()
print("landing:", os.listdir(f"{S.ROOT}/landing/labeled"), "pending:", os.listdir(f"{S.ROOT}/pending/labeled"))
