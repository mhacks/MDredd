from typing import ClassVar


class JudgingFailure(RuntimeError):
    code: ClassVar[str] = "JUDGING_FAILURE"


class JudgingNotStartedException(JudgingFailure):
    code: ClassVar[str] = "JUDGING_NOT_STARTED"

    def __init__(self, message: str = "Judging Has Not Started") -> None:
        super().__init__(message)


class JudgingAlreadyStartedException(JudgingFailure):
    code: ClassVar[str] = "JUDGING_ALREADY_STARTED"

    def __init__(self, message: str = "Judging Has Already Started") -> None:
        super().__init__(message)


class JudgingNeverStartedException(JudgingFailure):
    code: ClassVar[str] = "JUDGING_NEVER_STARTED"

    def __init__(self, message: str = "Judging Never Started") -> None:
        super().__init__(message)


class JudgeDoesNotOwnPairException(JudgingFailure):
    code: ClassVar[str] = "JUDGE_DOES_NOT_OWN_PAIR"

    def __init__(self, message: str = "This Judge does not own the pair!") -> None:
        super().__init__(message)


class IncorrectPairFormatException(JudgingFailure):
    code: ClassVar[str] = "INCORRECT_PAIR_FORMAT"

    def __init__(self, message: str = "This Judge did not submit a pair correctly!") -> None:
        super().__init__(message)
