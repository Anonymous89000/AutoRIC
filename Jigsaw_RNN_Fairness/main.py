
import util.fit_optimize
import util.data_process
import argparse
import json
from cal_fairness import add_assertion, add_solver, eval_acc
from json_parser import parse
import cvx_optimize

# n*20
params_index_lstm2_2 = [
    (2, 8),
    (2, 12),
    (2, 20),
    (2, 25),
    (2, 36),
]

params_index_lstm3_2 = [
    (3, 8),
    (3, 12),
    (3, 20),
    (3, 25),
    (3, 36),
]

# n*20
params_index_lstm2 = [
    (2, 1),
    (2, 2),
    (2, 7),
    (2, 11),
    (2, 18),
    (2, 24),
    (2, 33),
]
# n*20
params_index_lstm3 = [
    (3, 7),
    (3,11),
    (3,15),
    (3,19),
    (3,25),
    (3,28),
    (3,35)
]
# n*2
params_index_linear = [
    (4, 1),
    (4, 2)
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='nSolver')
    parser.add_argument('--spec', type=str, default='benchmark/rnn/nnet/jigsaw_lstm/spec.json',
                        help='the specification file')
    parser.add_argument('--algorithm', type=str, default='optimize',
                        help='the chosen algorithm')
    parser.add_argument('--threshold', type=float,
                        help='the threshold in sprt')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='the distance value')
    parser.add_argument('--dataset', type=str, default='jigsaw',
                        help='the data set for rnn experiments')
    params_index = params_index_lstm3

    args = parser.parse_args()
    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)
    eval_acc(model, 'jigsaw', assertion)
    net_name = "rnn_lstm3"
    import time
    start = time.time()
    cvx_optimize.cvx_optimize(net_name, model, assertion, solver, params_index, 0)
    end = time.time()
    print(f"{end - start:.2f}")