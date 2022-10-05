from typing import List, Tuple
from numba import njit
import numpy as np


@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@njit(cache=True)
def np_sum_axis1(x):
    """Numba compatible version of np.sum(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(x.shape[1]):
        out = np.sum(out, x[:, i])
    return out


@njit
def slice_doc(
    doc: np.ndarray, target: np.ndarray, windows: int, doc_size
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate slice of doc."""
    slices: List[Tuple[np.ndarray, np.ndarray]] = []
    for index in range(0, doc_size - windows + 2):
        slices.append((doc[index : index + windows], target[index : index + windows]))
    return slices


@njit(cache=True)
def perfomance_by_doc(
    individual: np.ndarray,
    doc: np.ndarray,
    target: np.ndarray,
    doc_size: int,
    individual_size: int,
    unknown_id: int,
):
    """Performance of individual on doc."""

    individual_data = individual[:individual_size].reshape(-1, 1)
    entity_mask: np.ndarray = individual_data > 0
    mask_unknown: np.ndarray = individual_data == unknown_id
    union_doc: int = 0
    intercep_doc: int = 0
    retrive_doc: int = 0

    if individual_size > doc_size + 2:
        return 0.0

    for sliced_doc, sliced_targed in slice_doc(doc, target, individual_size, doc_size):
        macth_tokens: np.ndarray = (sliced_doc == np.abs(individual_data)) | (
            mask_unknown
        )

        any_match = np_any_axis1(macth_tokens)
        total_match = np.sum(any_match)

        predict_span: np.ndarray = np.zeros(individual_size, dtype=np.bool8)
        if total_match == individual_size:
            # predict_span: np.ndarray = np_sum_axis1(macth_tokens * entity_mask)
            # np_sum_axis1(macth_tokens * entity_mask)
            for i in range(individual_size):
                predict_span[i] = np.sum(macth_tokens[i] * entity_mask[i])

        intercep_doc += (predict_span * sliced_targed.flatten()).sum()
        union_doc += (predict_span | sliced_targed.flatten()).sum()
        retrive_doc += predict_span.sum()
        if intercep_doc > retrive_doc:
            print(f"{intercep_doc} {retrive_doc}")

    if union_doc == 0:
        return -1.0

    if intercep_doc == 0:
        return 0.0

    if intercep_doc / retrive_doc > 1:
        print(intercep_doc / retrive_doc, intercep_doc, retrive_doc)

    return (intercep_doc / retrive_doc) * np.log2(intercep_doc)

    # return intercep_doc / retrive_doc * np.log2(individual_size)


@njit(cache=True)
def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    return (vector1 * vector2).sum() / (
        np.sqrt((vector1 * vector1).sum()) * np.sqrt((vector2 * vector2).sum())
    )
