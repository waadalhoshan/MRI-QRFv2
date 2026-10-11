# 2. Cohort definition and subject splits

## CDR-to-class mapping

| CDR | MRI-QRF class |
|---:|---|
| 0 | Non-Demented |
| 0.5 | Very Mild Dementia |
| 1 | Mild Dementia |
| 2 | Moderate Dementia |

## OASIS-1

Final labelled cohort: **235 subjects**

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Non-Demented | 135 | 95 | 20 | 20 |
| Very Mild Dementia | 70 | 49 | 10 | 11 |
| Mild Dementia | 28 | 20 | 4 | 4 |
| Moderate Dementia | 2 | 1 | 0 | 1 |
| **Total** | **235** | **165** | **34** | **36** |

OASIS-1 validation therefore legitimately contains **no Moderate Dementia subject**.

## OASIS-2

Final cohort: **150 subjects**

| Class | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Non-Demented | 73 | 51 | 11 | 11 |
| Very Mild Dementia | 53 | 37 | 8 | 8 |
| Mild Dementia | 21 | 15 | 3 | 3 |
| Moderate Dementia | 3 | 1 | 1 | 1 |
| **Total** | **150** | **104** | **23** | **23** |

## Complementary binary classification

The binary task reuses the **same subject-level partitions** and labels CDR=0 as negative and CDR>0 (0.5, 1, or 2) as positive.

| Dataset | Binary group | Train | Validation | Test |
|---|---|---:|---:|---:|
| OASIS-1 | CDR=0 | 95 | 20 | 20 |
| OASIS-1 | CDR>0 | 70 | 14 | 16 |
| OASIS-2 | CDR=0 | 51 | 11 | 11 |
| OASIS-2 | CDR>0 | 53 | 12 | 12 |

## Leakage prevention

The split is performed at the **subject level before slice extraction**. All 20 slices from one subject remain in one split.

Split seed: **2026**.

For exact replication of the exact subject IDs used in the manuscript, preserve and publish the non-image split manifests if permitted:

```text
data/manifests/oasis1_subject_split.csv
data/manifests/oasis2_subject_split.csv
```

If those manifests are unavailable, `split_subjects.py` reproduces the fixed class-level allocation and deterministic split procedure, but the manuscript's exact subject assignment should be verified against the original study manifest.
