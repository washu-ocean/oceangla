import numpy as np


def _fdr_correct_1d(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    pvals_ = pvals.copy()
    pvals_[np.isnan(pvals_)] = 1
    pvals_sortind = np.argsort(pvals_)
    pvals_sorted = np.take(pvals_, pvals_sortind)
    ecdffactor = np.arange(1, len(pvals_sorted) + 1) / float(len(pvals_sorted))
    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1
    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[pvals_sortind] = pvals_corrected
    del pvals_corrected
    return pvals_corrected_


def fdr_correct(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    FDR-correct a pval map
    """
    if pvals.ndim == 2:
        qvals = np.empty_like(pvals)
        for beta in range(pvals.shape[0]):
            qvals[beta, :] = _fdr_correct_1d(pvals[beta, :], alpha)
        return qvals
    elif pvals.ndim == 1:
        return _fdr_correct_1d(pvals)
    else:
        raise np.exceptions.AxisError(
            "We should only have 1 or 2 axes in our pvalue map."
        )
