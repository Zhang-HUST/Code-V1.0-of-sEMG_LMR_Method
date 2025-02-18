def get_params_of_cnnBlock_1(window: int):
    assert window in [64, 128, 192, 256]

    if window == 64:
        kernel_size_1, stride_1 = (1, 2), (1, 2)
    elif window == 128:
        kernel_size_1, stride_1 = (1, 4), (1, 4)
    elif window == 192:
        kernel_size_1, stride_1 = (1, 6), (1, 6)
    else:
        kernel_size_1, stride_1 = (1, 8), (1, 8)

    return kernel_size_1, stride_1


def get_params_of_cnnBlock_4(C: int):
    assert C in [8, 9]

    if C == 8:
        kernel_size_4, stride_4 = (2, 1), (2, 1)
    else:
        kernel_size_4, stride_4 = (3, 1), (3, 1)

    return kernel_size_4, stride_4


def get_params_of_cnnBlock_5(C: int):
    assert C in [8, 9]

    if C == 8:
        kernel_size_5 = (4, 1)
    else:
        kernel_size_5 = (3, 1)

    return kernel_size_5
