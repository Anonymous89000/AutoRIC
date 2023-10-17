import numpy as np
import ast

input = open('benchmark/rnn/nnet/jigsaw_lstm/original/model.txt', 'r')
lines = input.readlines()

print(len(lines))

for i in range(4):
    wline = 1 + 2 * i
    bline = 2 + 2 * i

    w = np.array(ast.literal_eval(lines[wline]))
    b = np.array(ast.literal_eval(lines[bline]))

    w = w.transpose(1,0)
    print(w.shape)

    wout = open('w' + str(i + 1) + '.txt', 'w')
    bout = open('b' + str(i + 1) + '.txt', 'w')

    wout.write(str(w.tolist()))
    bout.write(str(b.tolist()))

    wout.flush()
    bout.flush()

    wout.close()
    bout.close()

input.close()
