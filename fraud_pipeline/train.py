"""Train, evaluate and register the first model (becomes champion if there is none yet)."""
import modeling

modeling.setup_mlflow()
modeling.train_and_register()
