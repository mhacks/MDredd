
class JudgingNotStartedException(RuntimeError):

    def __init__(self, message="Judging Has Not Started"):
        self.message = message
        super().__init__(self.message)


class JudgingAlreadyStartedException(RuntimeError):

    def __init__(self, message="Judging Has Already Started"):
        self.message = message
        super().__init__(self.message)


class JudgeDoesNotOwnPairException(RuntimeError):

    def __init__(self, message="This Judge does not own the pair!"):
        self.message = message
        super().__init__(self.message)


class IncorrectPairFormatException(RuntimeError):

    def __init__(self, message="This Judge did not submit a pair correctly!"):
        self.message = message
        super().__init__(self.message)
