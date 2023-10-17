import torch
from deepZ.code import zonotope
from func_timeout import func_set_timeout
import func_timeout
import logging
DELTA = 10e-4


# 左闭右闭
# 确保右区间不鲁棒（否则要调整下算法惹）
def binarySearch(net: torch.nn.Module, inputs: torch.Tensor, true_label: int, l:float, r:float):
    while l <= r:
        mid = (l + r)/2
        # 鲁棒，向更大搜索
        res = False
        try:
            res= analyze(net, inputs, mid, true_label)
        except func_timeout.exceptions.FunctionTimedOut:
            logging.info(f"{mid}, time out")
            res = False
        if(res):
            l = mid + DELTA
        else:
            r = mid - DELTA
    return l

@func_set_timeout(120)
def analyze(net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:
    """ Analyze the robust eps
    """
    model = zonotope.Model(net, eps=eps, x=inputs, true_label=true_label)
    base_pred = net(inputs)
    del net

    print(f"[+] True label: {true_label}, Epsilon: {eps}")
    logging.debug(f"Base predictions: {base_pred[0]}")

    while not model.verify():
        model.updateParams()
    return True