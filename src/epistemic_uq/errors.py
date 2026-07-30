class EpistemicUQError(Exception):
    pass


class ConfigurationError(EpistemicUQError):
    pass


class BackendError(EpistemicUQError):
    pass


class ParsingError(EpistemicUQError):
    pass


class ValidationError(EpistemicUQError):
    pass


class StorageError(EpistemicUQError):
    pass


class CalibrationError(EpistemicUQError):
    pass
