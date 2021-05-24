from tqdm import tqdm, trange
import argparse
import numpy as np

from seminar.models import MFModel
from seminar.neural_collaborative_filtering.Dataset import Dataset
from seminar.neural_collaborative_filtering.evaluate import evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ds/ml-1m',
                    help='Path to the dataset')
parser.add_argument('--epochs', type=int, default=128,
                    help='Number of training epochs')
parser.add_argument('--embedding_dim', type=int, default=8,
                    help='Embedding dimensions, the first dimension will be '
                         'used for the bias.')
parser.add_argument('--regularization', type=float, default=0.0,
                    help='L2 regularization for user and item embeddings.')
parser.add_argument('--negatives', type=int, default=8,
                    help='Number of random negatives per positive examples.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='SGD step size.')
parser.add_argument('--stddev', type=float, default=0.1,
                    help='Standard deviation for initialization.')
args = parser.parse_args()

dataset = Dataset(args.data)
train, test_ratings, test_negatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives  # For MLP
# train = np.column_stack(dataset.trainMatrix.nonzero()) #For MF
# test_ratings, test_negatives = (dataset.testRatings, dataset.testNegatives)

# %%
# MLPModel(dataset.num_users, dataset.num_items)
model = MFModel(dataset.num_users, dataset.num_items,
                args.embedding_dim - 1, args.regularization, args.stddev)


def evaluate(model, test_ratings, test_negatives, K=10):
    (hits, ndcgs) = evaluate_model(model, test_ratings, test_negatives, K=K,
                                   num_thread=1)
    return np.array(hits).mean(), np.array(ndcgs).mean()


hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t' % (0, hr, ndcg))

for epoch in trange(args.epochs):
    _ = model.fit(train, learning_rate=args.learning_rate, num_negatives=args.negatives)
    hr, ndcg = evaluate(model, test_ratings, test_negatives, K=10)
    print('Epoch %4d:\t HR=%.4f, NDCG=%.4f\t' % (epoch + 1, hr, ndcg))
