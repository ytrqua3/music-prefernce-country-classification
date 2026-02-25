# script.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import joblib
import lightgbm as lgb
import io
import json
import awswrangler as wr


def model_fn(model_dir):
    """Load the LightGBM model"""
    print("✅ in model_fn")
    model_path = os.path.join(model_dir, "model.txt")
    print('✅successfully retrieved model')
    # or "model.joblib" if you used joblib
    model = lgb.Booster(model_file=model_path)
    le = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
    return model, le

def predict_fn(data, model_le):
    print("✅ in predict_fn")
    model = model_le[0]
    le = model_le[1]
    probs = model.predict(data)
    print('✅ successfully predicted')
    top5_idx = np.argsort(probs, axis=1)[0, -5:]
    preds = le.inverse_transform(top5_idx)
    output = {'countries': preds.tolist(), 'probabilities': probs[0, top5_idx].tolist()}
    print(f'final output: {output}')
    return output

def input_fn(request_body, request_content_type):
    print('✅ in input_fn')
    if request_content_type == 'application/x-npy':
        # Most common case: SageMaker Python SDK → binary .npy
        array = np.load(io.BytesIO(request_body), allow_pickle=True)
        if len(array) == 129:
            print("✅array with correct size detected")
            return array.reshape(1, -1)
        else:
            raise ValueError(f'Incorrect number of features provided, should be 129, not {len(input_data)}')
    elif request_content_type == 'application/json':
        user_top_artists = request_body.get('top_artists')
        user_total_scrobbles = request_body.get('total_scrobbles')
        print("✅Recieved json request")

        bucket = "music-preference-bucket"
        prefix = "artist_embeddings/"

        path = f"s3://{bucket}/{prefix}"
        artist_embeddings_df = wr.s3.read_parquet(
            path=path,
            dataset=True,               # ← this makes it read the whole dataset (all files + partitions)
            use_threads=True
        )
        print("✅Successfully fetched artist embeddings")
        print(type(artist_embeddings_df))
        print(type(artist_embeddings_df.loc['U2', 'vector']['values']))

        total_weight = 0
        user_embedding = np.zeros(128)
        for artist in user_top_artists:
            name = artist['artist_name']
            artist['weight'] = np.log10(int(artist['playcount'])+1)
            artist['emb'] = artist_embeddings_df.loc[name, 'vector']['values'].tolist()
            artist['scaled_emb'] = artist['emb'] * artist['weight']
            total_weight += artist['weight']
            user_embedding += artist['scaled_emb']
        user_embedding = user_embedding / total_weight
        user_embedding = user_embedding.tolist()
        user_embedding.append(np.log(user_total_scrobbles))
        user_embedding = np.array(user_embedding)
        print("✅Successfully computed user embeddings")
        print(user_embedding)
        return user_embedding.reshape(1, -1)
    else:
        raise ValueError(f'Incorrect request content type')


def output_fn(prediction, accept):
    if accept == 'application/json':
        print("✅dumping to output")
        print(prediction)#{'countries': ['Germany', 'Australia', 'Canada', 'United Kingdom', 'United States'], 'probabilities': [0.019101389136249886, 0.026567849518657885, 0.07321074981649198, 0.1832723202432694, 0.5797086470198316]}
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

if __name__ == '__main__':
    print("✅inside script.py, calling main...")
    main()
