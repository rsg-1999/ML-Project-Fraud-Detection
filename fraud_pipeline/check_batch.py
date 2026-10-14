"""Check the oldest arrived-but-unchecked batch with the champion; retrain if the rules in settings.py say so.
Optional: --batch batch_2 to re-check a specific one."""
import argparse

import lakehouse
import modeling

ap = argparse.ArgumentParser()
ap.add_argument("--batch")
batch = ap.parse_args().batch or lakehouse.next_unchecked()
if batch is None:
    print("no new batch")
else:
    modeling.setup_mlflow()
    modeling.check_batch(batch)
    lakehouse.mark_checked(batch)
