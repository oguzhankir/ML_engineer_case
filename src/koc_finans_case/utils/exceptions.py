class CustomException(Exception):
    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self):
        return f"{self.status_code}: {self.message}"


class ValidationException(CustomException):
    def __init__(self, message: str = "Validation Error"):
        super().__init__(message, 400)


class ModelNotLoadedException(CustomException):
    def __init__(self, message: str = "Model Not Loaded"):
        super().__init__(message, 500)


class PredictionException(CustomException):
    def __init__(self, message: str = "Prediction Error"):
        super().__init__(message, 500)


class PreprocessingException(CustomException):
    def __init__(self, message: str = "Preprocessing Error"):
        super().__init__(message, 500)


class ModelLoadingException(CustomException):
    def __init__(self, message: str = "Model Loading Error"):
        super().__init__(message, 500)


class HyperparameterOptimizationException(CustomException):
    def __init__(self, message: str = "Hyperparameter Optimization Error"):
        super().__init__(message, 500)


class FeatureSelectionException(CustomException):
    def __init__(self, message: str = "Feature Selection Error"):
        super().__init__(message, 500)


class ModelTrainingException(CustomException):
    def __init__(self, message: str = "Model Training Error"):
        super().__init__(message, 500)


class ConfigSavingException(CustomException):
    def __init__(self, message: str = "Configuration Saving Error"):
        super().__init__(message, 500)


class FeatureSelectionException(CustomException):
    def __init__(self, message: str = "Feature Selection Error"):
        super().__init__(message, 500)


class EncoderEvaluationException(CustomException):
    def __init__(self, message: str = "Encoder Evaluation Error"):
        super().__init__(message, 500)
