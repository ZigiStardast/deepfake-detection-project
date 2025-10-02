import os, argparse, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_embeddings_index(path):
    df = pd.read_csv(path)
    X, y = [], []
    for _, r in df.iterrows():
        emb_path = r["embedding_path"]
        if not os.path.exists(emb_path):
            continue
        emb = np.load(emb_path)
        X.append(emb)
        y.append(1 if r["label"] == "fake" else 0)
    return np.vstack(X), np.array(y)

def main(args):
    idx_csv = os.path.join(args.embeddings_dir, "embeddings_index.csv")
    print("Loading:", idx_csv)
    X, y = load_embeddings_index(idx_csv)
    print("Loaded embeddings:", X.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if args.classifier == "xgboost":
        if args.grid:
            model = XGBClassifier(eval_metric="logloss", n_jobs=4)
            param_grid = {
                "n_estimators": [200, 500],
                "max_depth": [3, 6],
                "learning_rate": [0.1],
            }
            clf = GridSearchCV(model, param_grid, cv=3, scoring="f1", verbose=1, n_jobs=2)
            clf.fit(X_train, y_train)
            best = clf.best_estimator_
            print("Best params:", clf.best_params_)
        else:
            best = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                n_jobs=4
            )
            best.fit(X_train, y_train)

    elif args.classifier == "rf":
        if args.grid:
            model = RandomForestClassifier(n_jobs=4)
            param_grid = {"n_estimators": [200, 500], "max_depth": [None, 30]}
            clf = GridSearchCV(model, param_grid, cv=3, scoring="f1", verbose=1, n_jobs=2)
            clf.fit(X_train, y_train)
            best = clf.best_estimator_
            print("Best params:", clf.best_params_)
        else:
            best = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=4)
            best.fit(X_train, y_train)

    elif args.classifier == "svm":
        if args.grid:
            model = SVC(probability=True)
            param_grid = {"C": [1, 10], "gamma": ["scale"]}
            clf = GridSearchCV(model, param_grid, cv=3, scoring="f1", verbose=1, n_jobs=2)
            clf.fit(X_train, y_train)
            best = clf.best_estimator_
            print("Best params:", clf.best_params_)
        else:
            best = SVC(C=1, gamma="scale", probability=True)
            best.fit(X_train, y_train)
    else:
        raise ValueError("Unknown classifier")

    # Evaluate
    y_pred = best.predict(X_val)
    y_prob = best.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, y_pred))
    print("ROC AUC:", roc_auc_score(y_val, y_prob))

    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.classifier}_model.joblib")
    joblib.dump(best, out_path)
    print("Saved model to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--classifier", choices=["xgboost", "rf", "svm"], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid", action="store_true", help="Use GridSearchCV (slow!)")
    args = p.parse_args()
    main(args)
