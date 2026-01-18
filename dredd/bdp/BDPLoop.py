import numpy as np
import scipy.special as special


class BDPLoop:
    """
    An implementation of Bayesian Decision Process for Cost-Efficient Dynamic Ranking via Crowdsourcing (Chen et al).
    """

    def __init__(self, K: int):
        self.K = K
        self.alpha_t = np.ones(K)

    def get_rankings(self):
        """
        Gets our current ranking in List[Project, score] format.
        """
        return np.argsort(self.alpha_t)[::-1]

    def get_next_pair(self):
        """
        Returns a (Project, Project) tuple that we should compare next.
        """
        best_score = float("-inf")
        best_pair = (0, 1)

        for i in range(self.K):
            for j in range(i + 1, self.K):
                score = self.compute_pair_score(i, j)
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)

        return best_pair

    def compute_pair_score(self, i: int, j: int) -> float:
        """
        Computes how much we think a given pair should be compared.
        What we're argmaxing over in eq. 31, p. 16.
        """
        denom = self.alpha_t[i] + self.alpha_t[j]
        if denom <= 0:
            return float("-inf")
        prob_i_wins = self.alpha_t[i] / denom
        return prob_i_wins * (self.R_tilde(i, j, +1) + self.R_tilde(i, j, -1))

    def submit_comparison(self, i: int, j: int, is_right_win: bool) -> None:
        """
        Updates our internal rankings given an observed comparison.
        Described in eq. 32, p.16.
        """
        Y_ij = 1 if is_right_win else -1
        self.alpha_t = self.MM(i, j, Y_ij)

    def R_tilde(self, i: int, j: int, Y_ij: int) -> float:  # vectorize
        """
        Computes the score r_tilde for a given pair (i, j) and outcome Y_ij.
        """
        alpha_hat = self.MM(i, j, Y_ij)

        sum_hat = 0.0
        for ip in range(self.K):
            for jp in range(ip + 1, self.K):
                a = np.minimum(alpha_hat[ip], alpha_hat[jp])
                b = np.maximum(alpha_hat[ip], alpha_hat[jp])
                sum_hat += special.betainc(a, b, 0.5)

        sum_alpha = 0.0
        for ip in range(self.K):
            for jp in range(ip + 1, self.K):
                a = np.minimum(self.alpha_t[ip], self.alpha_t[jp])
                b = np.maximum(self.alpha_t[ip], self.alpha_t[jp])
                sum_alpha += special.betainc(a, b, 0.5)

        r_tilde = (2.0 / (self.K * (self.K - 1))) * (sum_hat - sum_alpha)

        return r_tilde

    def MM(self, i: int, j: int, Y_ij: int) -> np.ndarray:
        """
        Moment matching update.
        Described in eq. 21, p. 13.
        """
        alpha_0 = np.sum(self.alpha_t)
        if alpha_0 <= 0:
            return self.alpha_t.copy()

        C = self.alpha_t / alpha_0
        C_ij_denom = alpha_0 * (self.alpha_t[i] + self.alpha_t[j] + 1.0)
        C[i] = (
            (self.alpha_t[i] + (1.0 + Y_ij) / 2.0) * (self.alpha_t[i] + self.alpha_t[j])
        ) / C_ij_denom
        C[j] = (
            (self.alpha_t[j] + (1.0 - Y_ij) / 2.0) * (self.alpha_t[i] + self.alpha_t[j])
        ) / C_ij_denom

        D_ij_denom = (
            alpha_0 * (alpha_0 + 1.0) * (self.alpha_t[i] + self.alpha_t[j] + 2.0)
        )
        D_i = (
            (self.alpha_t[i] + (1.0 + Y_ij) / 2.0)
            * (self.alpha_t[i] + (3.0 + Y_ij) / 2.0)
            * (self.alpha_t[i] + self.alpha_t[j])
        ) / D_ij_denom
        D_j = (
            (self.alpha_t[j] + (1.0 - Y_ij) / 2.0)
            * (self.alpha_t[j] + (3.0 - Y_ij) / 2.0)
            * (self.alpha_t[i] + self.alpha_t[j])
        ) / D_ij_denom

        D_rest_denom = alpha_0 * (alpha_0 + 1.0)
        D_all = np.sum(self.alpha_t * (self.alpha_t + 1.0)) / D_rest_denom
        D_extra = (
            self.alpha_t[i] * (self.alpha_t[i] + 1.0)
            + self.alpha_t[j] * (self.alpha_t[j] + 1.0)
        ) / D_rest_denom
        D_rest = D_all - D_extra
        D = D_i + D_j + D_rest

        sum_ck_sq = np.sum(C**2)
        numerator = D - 1.0
        denominator = sum_ck_sq - D

        if abs(denominator) < 1e-14:
            return self.alpha_t.copy()

        alpha_0_prime = numerator / denominator
        if alpha_0_prime <= 0:
            return self.alpha_t.copy()

        alpha_prime = C * alpha_0_prime
        return alpha_prime
