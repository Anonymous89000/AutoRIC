import autograd.numpy as np
import argparse
import json
import ast
import torch
from json_parser import parse
from utils import *


def add_assertion(args, spec):
    """ add assertion to spec.
    """
    assertion = {}

    assertion['robustness'] = 'local'
    assertion['distance'] = 'di'
    assertion['eps'] = str(args.eps)

    spec['assert'] = assertion


def add_solver(args, spec):
    """ add solver to spec.
    """
    solver = {}

    solver['algorithm'] = args.algorithm
    if args.algorithm == 'sprt':
        solver['threshold'] = str(args.threshold)
        solver['alpha'] = '0.05'
        solver['beta'] = '0.05'
        solver['delta'] = '0.005'

    spec['solver'] = solver

def eval_acc(model, dataset:str, assertion:dict):
    """ evaluate models' acc.
    """


    if(isinstance(model, str)):
        #is pt file
        model = torch.load(model)
    lower = model.lower[0]
    upper = model.upper[0]

    if dataset == 'jigsaw':
        pathX = 'benchmark/rnn/data/jigsaw/'
        pathY = 'benchmark/rnn/data/jigsaw/labels.txt'
    elif dataset == 'wiki':
        pathX = 'benchmark/rnn/data/wiki/'
        pathY = 'benchmark/rnn/data/wiki/labels.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    model.shape = np.asarray((100, 50))

    l_pass = 0
    l_fail = 0
    for i in range(100):
        assertion['x0'] = pathX + 'data' + str(i) + '.txt'
        x0 = np.array(ast.literal_eval(read(assertion['x0'])))

        shape_x0 = np.asarray((int(x0.size / 50), 50))

        model.shape = shape_x0
        model.lower = np.full(x0.size, lower)
        model.upper = np.full(x0.size, upper)

        output_x0 = model.apply(x0)
        lbl_x0 = np.argmax(output_x0, axis=1)[0]
        # print(f'Data {i}, y {y0s[i]}, lbl {lbl_x0}')

        # accuracy test

        if lbl_x0 == y0s[i]:
            l_pass = l_pass + 1
        else:
            l_fail = l_fail + 1

    print("Accuracy of ori network: %f.\n" % (l_pass / (l_pass + l_fail)))
    return l_pass / (l_pass + l_fail)

def cal_fairness_with_model(model, dataset: str, assertion:dict, solver:dict):
    """ Cal fairness for rnn lib model.
    """
    if(isinstance(model, str)):
        #is pt file
        model = torch.load(model)
        
    test_acc_only = True
    lower = model.lower[0]
    upper = model.upper[0]

    if dataset == 'jigsaw':
        pathX = 'benchmark/rnn/data/jigsaw/'
        pathY = 'benchmark/rnn/data/jigsaw/labels.txt'

    elif dataset == 'wiki':
        pathX = 'benchmark/rnn/data/wiki/'
        pathY = 'benchmark/rnn/data/wiki/labels.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    model.shape = np.asarray((100, 50))

    fair1 = 0
    fair2 = 0
    f1_l1 = 0
    f1_l0 = 0
    f2_l1 = 0
    f2_l0 = 0

    if test_acc_only:
        for i in range(50):
            assertion['x0'] = pathX + 'data' + str(i) + '.txt'
            # 数据文件
            x0 = np.array(ast.literal_eval(read(assertion['x0'])))
            # print(x0.shape)

            shape_x0 = np.asarray((int(x0.size / 50), 50))
            # print(shape_x0)
            shape_i = [1, 50]
            size_i = 50

            model.shape = shape_x0
            model.lower = np.full(x0.size, lower)
            model.upper = np.full(x0.size, upper)

            output_x0 = model.apply(x0)
            # print(output_x0.shape)
            lbl_x0 = 1 - np.argmax(output_x0, axis=1)[0]
            # print(lbl_x0)

            if (lbl_x0 == 1):
                f1_l1 = f1_l1 + 1

            if (lbl_x0 == 0):
                f1_l0 = f1_l0 + 1

            # print(f1_l1)
        fair1 = f1_l1 / (f1_l1 + f1_l0)

        for i in range(50, 100):
            assertion['x0'] = pathX + 'data' + str(i) + '.txt'
            # assertion['x0'] = pathX + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(assertion['x0'])))

            shape_x0 = np.asarray((int(x0.size / 50), 50))

            model.shape = shape_x0
            model.lower = np.full(x0.size, lower)
            model.upper = np.full(x0.size, upper)

            output_x0 = model.apply(x0)
            # print(output_x0.shape)
            lbl_x0 = 1 - np.argmax(output_x0, axis=1)[0]

            if (lbl_x0 == 1):
                f2_l1 = f2_l1 + 1
            if (lbl_x0 == 0):
                f2_l0 = f2_l0 + 1
        fair2 = f2_l1 / (f2_l1 + f2_l0)

        print(f"Fairness of ori network:{abs(fair1-fair2)}%.")
        return abs(fair1-fair2)
    else:
        solver.solve(model, assertion)

# 计算公平性


def cal_fairness():
    test_acc_only = True
    np.set_printoptions(threshold=20)
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

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)

    lower = model.lower[0]
    upper = model.upper[0]

    if args.dataset == 'jigsaw':
        pathX = 'benchmark/rnn/data/jigsaw/'
        pathY = 'benchmark/rnn/data/jigsaw/labels.txt'

    elif args.dataset == 'wiki':
        pathX = 'benchmark/rnn/data/wiki/'
        pathY = 'benchmark/rnn/data/wiki/labels.txt'

    y0s = np.array(ast.literal_eval(read(pathY)))

    model.shape = np.array([100, 50])

    l_pass1 = 0
    l_fail1 = 0
    l_pass2 = 0
    l_fail2 = 0
    l_pass = 0
    l_fail = 0
    fair1 = 0
    fair2 = 0
    f1_l1 = 0
    f1_l0 = 0
    f2_l1 = 0
    f2_l0 = 0

    if test_acc_only == True:
        for i in range(50):
            assertion['x0'] = pathX + 'data' + str(i) + '.txt'
            # assertion['x0'] = pathX + 'data_' + str(i) + '_' + str(j) + '.txt'
            # assertion['x0'] = pathX + 'data_' + str(i) + '_' + str(j) +'_' + str(j+1) + '.txt'
            # 数据文件
            x0 = np.array(ast.literal_eval(read(assertion['x0'])))
            # print(x0.shape)

            shape_x0 = (int(x0.size / 50), 50)
            # print(shape_x0)
            shape_i = [1, 50]
            size_i = 50

            model.shape = shape_x0
            model.lower = np.full(x0.size, lower)
            model.upper = np.full(x0.size, upper)

            output_x0 = model.apply(x0)
            # print(output_x0.shape)
            lbl_x0 = 1 - np.argmax(output_x0, axis=1)[0]
            # print(lbl_x0)

            if (lbl_x0 == 1):
                f1_l1 = f1_l1 + 1

            if (lbl_x0 == 0):
                f1_l0 = f1_l0 + 1

            # print(f1_l1)
        fair1 = f1_l1 / (f1_l1 + f1_l0)

        for i in range(50, 100):
            assertion['x0'] = pathX + 'data' + str(i) + '.txt'
            # assertion['x0'] = pathX + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(assertion['x0'])))
            # print(x0.shape)

            shape_x0 = (int(x0.size / 50), 50)
            # print(shape_x0)
            shape_i = [1, 50]
            size_i = 50

            model.shape = shape_x0
            model.lower = np.full(x0.size, lower)
            model.upper = np.full(x0.size, upper)

            output_x0 = model.apply(x0)
            # print(output_x0.shape)
            lbl_x0 = 1 - np.argmax(output_x0, axis=1)[0]

            if (lbl_x0 == 1):
                f2_l1 = f2_l1 + 1
            if (lbl_x0 == 0):
                f2_l0 = f2_l0 + 1
        fair2 = f2_l1 / (f2_l1 + f2_l0)
        # print(fair2)

        print("Fairness of ori network: %f.\n" % abs(fair1-fair2))
        return abs(fair1-fair2)
    else:
        solver.solve(model, assertion)

# 随机生成不同的数据用于计算公平性


def uniformsampling():
    pathX = 'benchmark/rnn/data/jigsaw/'
    # int j = 0
    for i in range(100):
        pathX_all = pathX + 'data' + str(i) + '.txt'
        # assertion['x0'] = pathX + str(i) + '.txt'
        x = np.array(ast.literal_eval(read(pathX_all)))
        size_i = 50
        len_x = int(x.size / size_i)
        # print(len_x)
        x0 = np.reshape(x, (len_x, 50))
        # print(x0.shape)
        for j in range(4):
            x1 = np.delete(x0, [j, j+1], axis=0)
            x1 = np.reshape(x1, -1)
            # print(x1.shape)
            x1 = x1.tolist()
            # print(x1)
            pathX_new = pathX + 'data_' + \
                str(i) + '_' + str(j) + '_' + str(j+1) + '.txt'
            # np.savetxt(pathX_new, x1, fmt = "%f")
            f = open(pathX_new, "w")
            f.write(str(x1))
            f.close()
        # print(x0.shape)


if __name__ == '__main__':

    # main()
    fair = cal_fairness()
    # uniformsampling()

# def main():
#     test_acc_only = True
#     np.set_printoptions(threshold=20)
#     parser = argparse.ArgumentParser(description='nSolver')

#     parser.add_argument('--spec', type=str, default='benchmark/rnn/nnet/jigsaw_lstm/spec.json',
#                         help='the specification file')
#     parser.add_argument('--algorithm', type=str,
#                         help='the chosen algorithm')
#     parser.add_argument('--threshold', type=float,
#                         help='the threshold in sprt')
#     parser.add_argument('--eps', type=float,
#                         help='the distance value')
#     parser.add_argument('--dataset', type=str,
#                         help='the data set for rnn experiments')

#     args = parser.parse_args()


#     with open(args.spec, 'r') as f:
#         spec = json.load(f)

#     add_assertion(args, spec)
#     add_solver(args, spec)

#     model, assertion, solver, display = parse(spec)

#     lower = model.lower[0]
#     upper = model.upper[0]

#     if args.dataset == 'jigsaw':
#         pathX = 'benchmark/rnn/data/jigsaw/'
#         pathY = 'benchmark/rnn/data/jigsaw/labels.txt'
#     elif args.dataset == 'wiki':
#         pathX = 'benchmark/rnn/data/wiki/'
#         pathY = 'benchmark/rnn/data/wiki/labels.txt'

#     y0s = np.array(ast.literal_eval(read(pathY)))

#     model.shape = (100, 50)

#     l_pass = 0
#     l_fail = 0

#     if test_acc_only == True:
#         for i in range(100):
#             assertion['x0'] = pathX + 'data' + str(i) + '.txt'
#             #assertion['x0'] = pathX + str(i) + '.txt'
#             x0 = np.array(ast.literal_eval(read(assertion['x0'])))

#             shape_x0 = (int(x0.size / 50), 50)

#             model.shape = shape_x0
#             model.lower = np.full(x0.size, lower)
#             model.upper = np.full(x0.size, upper)

#             output_x0 = model.apply(x0)
#             lbl_x0 = np.argmax(output_x0, axis=1)[0]
#             '''
#             lbl_x0 = 0
#             if output_x0[0][0] > output_x0[0][1]:
#                 lbl_x0 = 1
#             else:
#                 lbl_x0 = 0
#             '''
#             print('Data {}, y {}, lbl {}'.format(i, y0s[i], lbl_x0))

#             # accuracy test

#             if lbl_x0 == y0s[i]:
#                 l_pass = l_pass + 1
#             else:
#                 l_fail = l_fail + 1

#         print("Accuracy of ori network: %f.\n" % (l_pass / (l_pass + l_fail)))
#     else:
#         solver.solve(model, assertion)
