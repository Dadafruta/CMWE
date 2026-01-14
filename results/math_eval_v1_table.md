# v2_holdout summary

## per-mode metrics

| mode | rows | routes | empty_rate | unk_any_rate | unique_out_rate | len_min | len_median | len_max | hit_max_len_rate | unanswerable_rate | refusal_on_unanswerables | false_refusal_on_answerables | acc_answerables | coverage_answerables | acc_when_answered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_like | 200 | {} | 0.0000 | 0.0000 | 0.4500 | 71 | 194.0 | 671 | 0.0050 | NA | NA | NA | NA | NA | NA |
| cmwe | 200 | {} | 0.0000 | 0.0000 | 0.4500 | 71 | 194.0 | 671 | 0.0050 | NA | NA | NA | NA | NA | NA |
| always_guard | 200 | {} | 0.0000 | 0.0000 | 0.4500 | 71 | 194.0 | 671 | 0.0050 | NA | NA | NA | NA | NA | NA |

## cross-mode comparisons

| a | b | rows_compared | pct_same_out | pct_same_refused | pct_same_correct | pct_same_route |
| --- | --- | --- | --- | --- | --- | --- |
| base_like | cmwe | 200 | 1.0000 | NA | NA | NA |
| base_like | always_guard | 200 | 1.0000 | NA | NA | NA |
