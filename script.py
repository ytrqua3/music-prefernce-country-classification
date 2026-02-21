# script.py
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import joblib
import io

def model_fn(model_dir):
    """Load the LightGBM model"""
    model_path = os.path.join(model_dir, "model.txt")
    # or "model.joblib" if you used joblib
    model = lgb.Booster(model_file=model_path)
    le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    return model, le

def predict_fn(data, model_le):
    model = model_le[0]
    le = model_le[1]
    probs = model.predict(data)
    return le.inverse_transform(np.argsort(probs, axis=1)[0, -5:])

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/x-npy':
        # Most common case: SageMaker Python SDK → binary .npy
        array = np.load(io.BytesIO(request_body), allow_pickle=False)
        if len(array) == 129:
            return array
        else:
            raise ValueError(f'Incorrect number of features provided, should be 129, not {len(input_data)}')
    else:
        raise ValueError(f'Incorrect request content type')


def output_fn(prediction, accept):
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"unsupported accept type: {accept}")

def main():
    #parse arguments
    print("✅fetching command line arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('--early_stopping_rounds', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=2.0)
    parser.add_argument('--num_leaves', type=int, default=64)
    parser.add_argument('--max_depth', type=int, default=-1)
    parser.add_argument('--min_data_in_leaf', type=int, default=100)
    parser.add_argument('--training_path', type=str)
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args = parser.parse_args()

    # Load training data
    print("✅loading training data")
    train_df = pd.read_parquet(args.training_path)

    # Assume last column is the target
    le = LabelEncoder()

    train_X = train_df.drop(['country', 'user_id'], axis=1)
    train_y = le.fit_transform(train_df.iloc[:, 0])
    print(train_X.shape)
    print(train_y.shape)

    # LightGBM dataset
    train_data = lgb.Dataset(train_X, label=train_y)

    # LightGBM parameters
    params = {
        "objective": "multiclass",
        "num_class": len(le.classes_),

        # Boosting
        "learning_rate": args.learning_rate,

        # Tree capacity (higher than default — you have data)
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,

        # CRITICAL for long-tail countries
        "min_data_in_leaf": args.min_data_in_leaf,

        # Regularization via randomness
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,

        # Stability
        "lambda_l2": args.l2,

        # Metrics
        "metric": "multi_logloss",
        "verbosity": -1
    }
    print("✅start training")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds),
        ]
    )

    # Save model
    print("✅start saving model")
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, 'model.txt')
    model.save_model(model_path)
    le_path = os.path.join(args.model_dir, 'label_encoder.joblib')
    joblib.dump(le, le_path)
    print(f"✅Model saved to {model_path}")

if __name__ == '__main__':
    print("✅inside script.py, calling main...")
    main()
