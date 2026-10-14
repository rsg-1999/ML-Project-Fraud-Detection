"""Score the unlabeled clients with the champion, save predictions, log label-free monitoring metrics."""
import modeling

modeling.setup_mlflow()
preds = modeling.score_unlabeled()
print(f"scored {len(preds)} clients, flagged {preds.flag.mean():.1%}")
