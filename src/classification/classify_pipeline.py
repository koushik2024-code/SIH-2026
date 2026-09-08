"""
Classification Pipeline -- End-to-End Orchestrator.

Chains model training, evaluation, model selection, persistence,
and inference into a single cohesive pipeline.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from config import settings
from .model_trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .inference import InferencePipeline

logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """
    End-to-end classification pipeline for fire type prediction.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        cv_folds: int = 5,
        n_iter: int = 50,
        search_method: str = "random",
        random_state: int = 42,
    ):
        self.trainer = ModelTrainer(
            test_size=test_size,
            cv_folds=cv_folds,
            n_iter=n_iter,
            search_method=search_method,
            random_state=random_state,
        )
        self.evaluator = None  # Initialized after training
        self.inference = InferencePipeline()

        self._models_dir = settings.BASE_DIR / "models"
        self._output_dir = settings.PROCESSED_DATA_DIR

    # ------------------------------------------------------------------
    # Training Pipeline
    # ------------------------------------------------------------------

    def run_training_pipeline(self) -> dict:
        """
        Execute the full model training and evaluation pipeline.

        Steps:
            1. Load training_dataset.csv from Part 3
            2. Split features / labels
            3. Train RF + XGBoost with hyperparameter tuning
            4. Evaluate both on held-out test set
            5. Select best model by weighted F1
            6. Save models + metrics + plots
        """
        logger.info("=" * 65)
        logger.info("  CLASSIFICATION PIPELINE -- TRAINING MODE")
        logger.info("=" * 65)

        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ==============================================================
        # Step 1: Load training data
        # ==============================================================
        training_path = self._output_dir / "training_dataset.csv"
        if not training_path.exists():
            logger.error(
                f"Training dataset not found at {training_path}. "
                "Run Part 3 first (python main.py --part 3)."
            )
            return {}

        df = pd.read_csv(training_path)
        logger.info(f"Loaded training dataset: {len(df)} records, {len(df.columns)} columns")

        if "fire_label" not in df.columns:
            logger.error("No 'fire_label' column in training dataset.")
            return {}

        # ==============================================================
        # Step 2: Prepare X and y
        # ==============================================================
        feature_cols = [c for c in df.columns if c not in ["fire_label", "fire_class_id"]]
        X = df[feature_cols]
        y = df["fire_label"]

        logger.info(f"Features ({len(feature_cols)}): {feature_cols}")
        logger.info(f"Label distribution:\n{y.value_counts().to_string()}")

        # ==============================================================
        # Step 3: Train models
        # ==============================================================
        train_results = self.trainer.train(X, y)

        class_names = self.trainer.get_class_names()
        feature_names = self.trainer.get_feature_names()

        # ==============================================================
        # Step 4: Evaluate models
        # ==============================================================
        self.evaluator = ModelEvaluator(class_names=class_names)
        eval_results = {}

        for name, res in train_results.items():
            model = res["model"]
            X_test = res["X_test"]
            y_test = res["y_test"]

            model_label = "Random Forest" if name == "rf" else "XGBoost"
            eval_result = self.evaluator.evaluate(model, X_test, y_test, model_name=model_label)
            eval_results[name] = eval_result

            # Confusion matrix
            cm_path = self._output_dir / f"confusion_matrix_{name}.png"
            self.evaluator.plot_confusion_matrix(model, X_test, y_test, cm_path, model_label)

            # ROC curves
            roc_path = self._output_dir / f"roc_curves_{name}.png"
            self.evaluator.plot_roc_curves(model, X_test, y_test, roc_path, model_label)

        # ==============================================================
        # Step 5: Compare and select best model
        # ==============================================================
        best_model_key = self.evaluator.compare_models(eval_results)

        # Model comparison chart
        compare_path = self._output_dir / "model_comparison.png"
        self.evaluator.plot_model_comparison(eval_results, compare_path)

        # ==============================================================
        # Step 6: Save models and metadata
        # ==============================================================
        # Save all models
        for name, res in train_results.items():
            model_filename = f"{'random_forest' if name == 'rf' else 'xgboost'}.pkl"
            model_path = self._models_dir / model_filename
            self.trainer.save_model(res["model"], model_path)

        # Save best model separately
        best_model = train_results[best_model_key]["model"]
        best_path = self._models_dir / "best_model.pkl"
        self.trainer.save_model(best_model, best_path)

        # Save label encoder
        le_path = self._models_dir / "label_encoder.pkl"
        self.trainer.save_model(self.trainer.get_label_encoder(), le_path)

        # Build class_id_map from label encoder
        le = self.trainer.get_label_encoder()
        class_id_map = {name: int(i) for i, name in enumerate(le.classes_)}

        # Model metadata
        metadata = {
            "best_model": best_model_key,
            "best_model_label": "Random Forest" if best_model_key == "rf" else "XGBoost",
            "best_model_path": str(best_path),
            "feature_names": feature_names,
            "class_names": class_names,
            "class_id_map": class_id_map,
            "model_files": {
                name: str(self._models_dir / f"{'random_forest' if name == 'rf' else 'xgboost'}.pkl")
                for name in train_results
            },
            "hyperparameters": {
                name: {k: str(v) for k, v in res.get("best_params", {}).items()}
                for name, res in train_results.items()
            },
            "cv_scores": {
                name: round(float(res.get("cv_score", 0)), 4)
                for name, res in train_results.items()
            },
        }

        metadata_path = self._models_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        logger.info(f"Model metadata saved -> {metadata_path}")

        # Save evaluation results
        eval_path = self._output_dir / "evaluation_metrics.json"
        self.evaluator.export_results(eval_results, eval_path)

        # ==============================================================
        # Summary
        # ==============================================================
        logger.info("=" * 65)
        logger.info("  CLASSIFICATION TRAINING COMPLETE")
        logger.info("=" * 65)
        logger.info(f"  Best model:        {metadata['best_model_label']}")
        logger.info(f"  CV F1 (weighted):  {metadata['cv_scores'].get(best_model_key, 'N/A')}")
        logger.info(f"  Test F1 (weighted): {eval_results[best_model_key]['overall_metrics'].get('f1_weighted', 'N/A')}")
        logger.info(f"  Model files:")
        logger.info(f"    {best_path}")
        logger.info(f"    {metadata_path}")
        logger.info(f"    {eval_path}")
        logger.info("=" * 65)

        return eval_results

    # ------------------------------------------------------------------
    # Inference Pipeline
    # ------------------------------------------------------------------

    def run_inference_pipeline(self, input_path: str = None) -> pd.DataFrame:
        """
        Run inference on new or existing fire data.

        Parameters
        ----------
        input_path : str, optional
            Path to input CSV. Defaults to enriched_fire_data.csv.

        Returns
        -------
        DataFrame
            Fire data with predictions.
        """
        logger.info("=" * 65)
        logger.info("  CLASSIFICATION PIPELINE -- INFERENCE MODE")
        logger.info("=" * 65)

        # Load model
        model_path = self._models_dir / "best_model.pkl"
        metadata_path = self._models_dir / "model_metadata.json"

        if not model_path.exists():
            logger.error(
                f"No trained model found at {model_path}. "
                "Run training first (python main.py --part 3.5)."
            )
            return pd.DataFrame()

        self.inference.load(str(model_path), str(metadata_path))

        # Load input data
        if input_path:
            data_path = Path(input_path)
        else:
            # Default: use enriched fire data
            data_path = self._output_dir / "enriched_fire_data.csv"

        if not data_path.exists():
            logger.error(f"Input data not found: {data_path}")
            return pd.DataFrame()

        fire_df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(fire_df)} records from {data_path}")

        # Run predictions
        classified_df = self.inference.predict(fire_df)

        # Save results
        output_path = self._output_dir / "classified_fires.csv"
        classified_df.to_csv(output_path, index=False)
        logger.info(f"Classified results saved -> {output_path}")

        # Summary
        logger.info("=" * 65)
        logger.info("  INFERENCE COMPLETE")
        logger.info("=" * 65)
        logger.info(f"  Records classified: {len(classified_df)}")
        logger.info(f"  Output: {output_path}")
        if "prediction_confidence" in classified_df.columns:
            mean_conf = classified_df["prediction_confidence"].mean()
            logger.info(f"  Mean confidence:    {mean_conf:.4f}")
        logger.info("=" * 65)

        return classified_df
