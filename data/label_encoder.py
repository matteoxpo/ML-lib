class LabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}
        self.is_fitted = False

    def fit(self, y):
        """
        Обучает кодировщик на метках.

        Параметры:
        y : array-like
            Массив меток для обучения.
        """
        unique_labels = set(y)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.is_fitted = True

    def transform(self, y):
        """
        Преобразует метки в числовые значения.

        Параметры:
        y : array-like
            Массив меток для преобразования.

        Возвращает:
        array-like
            Преобразованные метки.
        """
        if not self.is_fitted:
            raise ValueError("MyLabelEncoder не обучен. Вызовите метод 'fit' перед 'transform'.")
        
        return [self.label_to_index[label] for label in y]

    def inverse_transform(self, y_encoded):
        """
        Преобразует числовые значения обратно в метки.

        Параметры:
        y_encoded : array-like
            Массив числовых значений для преобразования.

        Возвращает:
        array-like
            Декодированные метки.
        """
        return [self.index_to_label[idx] for idx in y_encoded]

    def fit_transform(self, y):
        """
        Обучает кодировщик и преобразует метки в числовые значения.

        Параметры:
        y : array-like
            Массив меток для обучения и преобразования.

        Возвращает:
        array-like
            Преобразованные метки.
        """
        self.fit(y)
        return self.transform(y)
