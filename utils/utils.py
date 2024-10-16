def check_size(y_true, y_pred):
    length = len(y_true)
    if length != len(y_pred):
        raise Exception("Wrong size")
    return length