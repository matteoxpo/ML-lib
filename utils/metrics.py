# utils/metrics.py

import numpy as np

from .utils import check_size

def accuracy(y_true, y_pred):
    """
    Рассчитывает точность (accuracy) классификатора.
    
    Параметры:
    y_true : array-like
        Истинные метки классов.
    y_pred : array-like
        Предсказанные метки классов.
    
    Возвращает:
    float
        Значение точности.
    """
    length = check_size(y_true, y_pred)
    if length == 0:
        return 0.0 
    
    right_answer_count = np.sum(y_true == y_pred)  
    return right_answer_count / length
        

def calc_tp_fp_fn_tn(y_true, y_pred):
    """
    Рассчитывает количество истинно положительных, ложноположительных, ложноотрицательных и истинно отрицательных предсказаний.
    
    Параметры:
    y_true : array-like
        Истинные метки классов.
    y_pred : array-like
        Предсказанные метки классов.
        
    Возвращает:
    tuple
        (TP, FP, FN, TN)
    """
    length = check_size(y_true, y_pred)
    tp = fp = fn = tn = 0

    for i in range(length):
        if y_true[i] == 1 and y_pred[i] == 1:  # TP
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:  # FN
            fn += 1
        elif y_true[i] == 0 and y_pred[i] == 1:  # FP
            fp += 1
        elif y_true[i] == 0 and y_pred[i] == 0:  # TN
            tn += 1
            
    return tp, fp, fn, tn
        
    

def precision(y_true, y_pred):
    """
    Рассчитывает точность (precision) классификатора.
    
    Параметры:
    y_true : array-like
        Истинные метки классов.
    y_pred : array-like
        Предсказанные метки классов.
    
    Возвращает:
    float
        Значение точности.
    """
    tp, fp, fn, tn = calc_tp_fp_fn_tn(y_true, y_pred)
    
    if (tp + fp) == 0:
        return 0.0
    
    return tp / (tp + fp)    


def recall(y_true, y_pred):
    """
    Рассчитывает полноту (recall) классификатора.
    
    Параметры:
    y_true : array-like
        Истинные метки классов.
    y_pred : array-like
        Предсказанные метки классов.
    
    Возвращает:
    float
        Значение полноты.
    """
    tp, fp, fn, tn = calc_tp_fp_fn_tn(y_true, y_pred)
    
    if (tp + fn) == 0:
        return 0.0
    
    return tp / (tp + fn) 


def f1_score(y_true, y_pred):
    """
    Рассчитывает F1-меру (F1 score) классификатора.
    
    Параметры:
    y_true : array-like
        Истинные метки классов.
    y_pred : array-like
        Предсказанные метки классов.
    
    Возвращает:
    float
        Значение F1-меры.
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    
    if precision_val + recall_val == 0:
        return 0.0
    
    return (2 * precision_val * recall_val) / (precision_val + recall_val)


def mean_squared_error(y_true, y_pred):
    """
    Рассчитывает среднеквадратичную ошибку (Mean Squared Error) для регрессии.
    
    Параметры:
    y_true : array-like
        Истинные значения.
    y_pred : array-like
        Предсказанные значения.
    
    Возвращает:
    float
        Значение среднеквадратичной ошибки.
    """
    
    length = check_size(y_true, y_pred)
    if length == 0:
        return 0.0
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Рассчитывает корень из среднеквадратичной ошибки (Root Mean Squared Error).
    
    Параметры:
    y_true : array-like
        Истинные значения.
    y_pred : array-like
        Предсказанные значения.
    
    Возвращает:
    float
        Значение корня из среднеквадратичной ошибки.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    Рассчитывает коэффициент детерминации (R² score) для регрессии.
    
    Параметры:
    y_true : array-like
        Истинные значения.
    y_pred : array-like
        Предсказанные значения.
    
    Возвращает:
    float
        Значение R².
    """
    if len(y_true) == 0:
        return 0.0 
    
    SSres = np.sum((y_pred - y_true) ** 2)
    SStot = np.sum( (y_true - np.mean(y_true)) ** 2)
    
    return 1 - SSres / SStot

def roc_auc_score(y_true, y_scores):
    """
    Рассчитывает AUC (Area Under the Curve) для ROC-кривой.
    
    Параметры:
    y_true : array-like
        Истинные метки классов (0 или 1).
    y_scores : array-like
        Вероятности положительного класса.
    
    Возвращает:
    float
        Значение AUC.
    """
    thresholds = np.unique(y_scores)
    trps = []
    fprs = []
    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in y_scores]
        tp, fp, fn, tn = calc_tp_fp_fn_tn(y_true, y_pred)
        trp = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        trps.append(trp)
        fprs.append(fpr)
    
    sorted_indices = np.argsort(fprs)
    trps = trps[sorted_indices]
    fprs = fprs[sorted_indices]
    
    # ось х - FPR ось y - TRP
    auc = 0
    for i in range(len(trps) - 1):
        auc += (fprs[i+1] - fprs[i]) * (trps[i + 1] + trps[i]) * 0.5
    # auc = np.trapz(trps, fprs)
    return auc 


def log_loss(y_true, y_pred):
    """
    Рассчитывает логарифмическую потерю (Log Loss) для бинарной классификации.
    
    Параметры:
    y_true : array-like
        Истинные метки классов (0 или 1).
    y_pred : array-like
        Предсказанные вероятности положительного класса.
    
    Возвращает:
    float
        Значение логарифмической потери.
    """
    N = check_size(y_true, y_pred)
    
    # noize
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    log_loss = - 1 / N * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return log_loss


def confusion_matrix(y_true, y_pred):
    """
    Создает матрицу ошибок (confusion matrix).
    
    Параметры:
    y_true : array-like
        Истинные метки классов.
    y_pred : array-like
        Предсказанные метки классов.
    
    Возвращает:
    array-like
        Матрица ошибок.
    """
    tp, fp, fn, tn = calc_tp_fp_fn_tn(y_true, y_pred)
    return [[tn, fp],
            [fn, tp]]

def mean_absolute_error(y_true, y_pred):
    """
    Рассчитывает среднюю абсолютную ошибку (Mean Absolute Error) для регрессии.
    
    Параметры:
    y_true : array-like
        Истинные значения.
    y_pred : array-like
        Предсказанные значения.
    
    Возвращает:
    float
        Значение средней абсолютной ошибки.
    """
    length = check_size(y_true, y_pred)
    if length == 0:
        return 0.0
    return np.mean(np.abs(y_true - y_pred))


def explained_variance_score(y_true, y_pred):
    """
    Рассчитывает коэффициент объясненной дисперсии (Explained Variance Score).
    
    Параметры:
    y_true : array-like
        Истинные значения.
    y_pred : array-like
        Предсказанные значения.
    
    Возвращает:
    float
        Значение объясненной дисперсии.
    """
    
    """
    explained_variance_score (коэффициент объясненной дисперсии) — 
    метрика, используемая для оценки качества регрессионных моделей. 
    Она измеряет, какую долю дисперсии зависимой переменной (целевой переменной) 
    объясняет модель. Эта метрика может варьироваться от 0 до 1, где 1 означает, 
    что модель идеально предсказывает целевую переменную, а 0 указывает на то, 
    что модель не объясняет дисперсию целевой переменной лучше, чем простое среднее значение.
    """
    check_size(y_true, y_pred)
    
    var_y_true = np.var(y_true)
    if var_y_true == 0:
        raise ValueError("Дисперсия истинных значений равна нулю.")
    
    return 1 - (np.var(y_true - y_pred) / var_y_true)
