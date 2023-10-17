#给定待修改参数位置以及扰动 给出公平性取值
#输出为csv文件
def getpara2(model, target, params_index):
    """ get modified output weights of neuron and the 1/robustness.
    """
    p = [] 
    count = 0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        tmp_matrix = model[f'layers.{layer_index}.weight'].T
        for j in range(0, len(tmp_matrix[neuron_index])):
            #print(tmp_matrix.shape)
            p.append(float(tmp_matrix[neuron_index][j]))
            count += 1
    p.append(target)
    return p


def modify(model, params, disturb):
    """ modify the neuron weights.
    """
    for param in params:
        layer, pos = param[0], param[1] - 1
        weight = model[f'layers.{layer}.weight'].T
        weight[pos] += disturb
        model[f'layers.{layer}.weight'] = weight.T

    return  model


















