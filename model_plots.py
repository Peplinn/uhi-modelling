import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def plot_ensemble_learning_curves(tuned_models, X_train, y_train, X_test, y_test, targets, max_trees=300, step=10):
    """
    Plots learning curves for tree-based models (Gradient Boosting, XGBoost, Random Forest)
    Shows R2 per target vs number of trees.
    
    Parameters:
    - tuned_models: dict of trained MultiOutputRegressor models
    - X_train, y_train, X_test, y_test: datasets
    - targets: list of target names
    - max_trees: maximum number of trees to plot
    - step: interval of trees to evaluate
    """
    for model_name, model in tuned_models.items():
        # Only consider tree-based models
        if model_name not in ["Gradient Boosting", "XGBoost", "Random Forest"]:
            continue

        plt.figure(figsize=(10, 6))
        plt.title(f"{model_name} Learning Curves for All Targets")
        plt.xlabel("Number of Trees / Iterations")
        plt.ylabel("R2 Score")

        for i, target_name in enumerate(targets):
            estimator = model.estimators_[i]
            train_scores = []
            test_scores = []

            if model_name == "Gradient Boosting":
                # true staged_predict
                for y_pred_train, y_pred_test in zip(
                    estimator.staged_predict(X_train),
                    estimator.staged_predict(X_test)
                ):
                    train_scores.append(r2_score(y_train.iloc[:, i], y_pred_train))
                    test_scores.append(r2_score(y_test.iloc[:, i], y_pred_test))

            elif model_name == "XGBoost":
                # staged predictions using iteration_range
                n_estimators = estimator.n_estimators
                for n in range(1, n_estimators + 1, step):
                    y_pred_train = estimator.predict(X_train, iteration_range=(0, n))
                    y_pred_test = estimator.predict(X_test, iteration_range=(0, n))
                    train_scores.append(r2_score(y_train.iloc[:, i], y_pred_train))
                    test_scores.append(r2_score(y_test.iloc[:, i], y_pred_test))

            elif model_name == "Random Forest":
                # Get best params from trained estimator
                best_params = estimator.get_params()

                # Remove things we shouldn't carry over
                best_params.pop("n_estimators", None)
                best_params["warm_start"] = True

                rf = RandomForestRegressor(**best_params)

                train_scores = []
                test_scores = []

                for n in range(1, max_trees + 1, step):
                    rf.set_params(n_estimators=n)
                    rf.fit(X_train, y_train.iloc[:, i])

                    y_pred_train = rf.predict(X_train)
                    y_pred_test = rf.predict(X_test)

                    train_scores.append(r2_score(y_train.iloc[:, i], y_pred_train))
                    test_scores.append(r2_score(y_test.iloc[:, i], y_pred_test))

            # Plot lines
            x_values = range(1, len(train_scores) + 1) if model_name == "Gradient Boosting" else (
                np.arange(step, step * len(train_scores) + 1, step) if model_name != "Gradient Boosting" else range(1, len(train_scores) + 1)
            )
            plt.plot(x_values, train_scores, linestyle='--', label=f"{target_name} Train")
            plt.plot(x_values, test_scores, linestyle='-', label=f"{target_name} Test")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()