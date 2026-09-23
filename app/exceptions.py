from typing import ClassVar


class JudgingFailure(RuntimeError):
    code: ClassVar[str] = "JUDGING_FAILURE"

    def __init__(self) -> None:
        super().__init__(self.code)


class JudgingNotStartedException(JudgingFailure):
    code: ClassVar[str] = "JUDGING_NOT_STARTED"


class JudgingAlreadyStartedException(JudgingFailure):
    code: ClassVar[str] = "JUDGING_ALREADY_STARTED"


class JudgingNeverStartedException(JudgingFailure):
    code: ClassVar[str] = "JUDGING_NEVER_STARTED"


class JudgeDoesNotOwnPairException(JudgingFailure):
    code: ClassVar[str] = "JUDGE_DOES_NOT_OWN_PAIR"


class IncorrectPairFormatException(JudgingFailure):
    code: ClassVar[str] = "INCORRECT_PAIR_FORMAT"


class UnknownRowException(JudgingFailure):
    code: ClassVar[str] = "UNKNOWN_ROW"


class InvalidColumnsException(JudgingFailure):
    code: ClassVar[str] = "INVALID_COLUMNS"

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__()
