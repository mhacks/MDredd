class JudgingNotStartedException(RuntimeError):
    def __init__(self, message: str = "Judging Has Not Started") -> None:
        super().__init__(message)


class JudgingAlreadyStartedException(RuntimeError):
    def __init__(self, message: str = "Judging Has Already Started") -> None:
        super().__init__(message)


class JudgingNeverStartedException(RuntimeError):
    def __init__(self, message: str = "Judging Never Started") -> None:
        super().__init__(message)


class JudgeDoesNotOwnPairException(RuntimeError):
    def __init__(self, message: str = "This Judge does not own the pair!") -> None:
        super().__init__(message)


class IncorrectPairFormatException(RuntimeError):
    def __init__(self, message: str = "This Judge did not submit a pair correctly!") -> None:
        super().__init__(message)
