# CPU wall-clock (torch CPU backend, 3 seeds)

| dataset | k | method | train MSE | time (s) | passes |
|---|---|---|---|---|---|
| GloVe-100 | 16,384 | e-means T5 (torch cpu) | 0.547438 ± 3.274e-05 | 23.3 ± 0.5 | 5.00 |
| GloVe-100 | 16,384 | e-means official (faiss cpu) T5 | 0.547465 ± 2.689e-05 | 90.9 ± 0.3 | 5.00 |
| GloVe-100 | 16,384 | h-means (EMA 0.4) T3 (torch cpu) | 0.548876 ± 2.376e-05 | 16.1 ± 2.8 | 3.10 |
| GloVe-100 | 16,384 | h-means (EMA 0.4) T5 (torch cpu) | 0.546176 ± 3.043e-05 | 25.0 ± 3.1 | 5.10 |
| GloVe-100 | 16,384 | h-means (EMA 0.8) T3 (torch cpu) | 0.548275 ± 3.092e-05 | 15.9 ± 0.5 | 3.10 |
| GloVe-100 | 16,384 | h-means (EMA 0.8) T5 (torch cpu) | 0.54553 ± 4.469e-05 | 22.4 ± 0.8 | 5.10 |
| GloVe-100 | 16,384 | lloyd@25 (torch cpu) | 0.552574 ± 6.713e-05 | 118.8 ± 3.7 | 25.00 |
| GloVe-100 | 65,536 | e-means T5 (torch cpu) | 0.480463 ± 9.037e-05 | 98.8 ± 8.1 | 5.00 |
| GloVe-100 | 65,536 | e-means official (faiss cpu) T5 | 0.480476 ± 9.852e-05 | 324.5 ± 14.6 | 5.00 |
| GloVe-100 | 65,536 | h-means (EMA 0.8) T3 (torch cpu) | 0.472994 ± 3.013e-05 | 49.9 ± 3.9 | 3.10 |
| GloVe-100 | 65,536 | h-means (EMA 0.8) T5 (torch cpu) | 0.47103 ± 5.474e-05 | 77.1 ± 2.6 | 5.10 |
| GloVe-100 | 65,536 | h-means T3 (torch cpu) | 0.471999 ± 3.099e-05 | 59.0 ± 1.6 | 3.10 |
| GloVe-100 | 65,536 | h-means T5 (torch cpu) | 0.468068 ± 3.087e-05 | 92.4 ± 3.1 | 5.10 |
| SIFT-128 | 16,384 | e-means T5 (torch cpu) | 37829.5 ± 0.8657 | 19.6 ± 1.5 | 5.00 |
| SIFT-128 | 16,384 | e-means official (faiss cpu) T5 | 37829.9 ± 2.999 | 76.8 ± 0.3 | 5.00 |
| SIFT-128 | 16,384 | h-means (EMA 0.4) T3 (torch cpu) | 38049.1 ± 2.251 | 12.0 ± 1.5 | 3.10 |
| SIFT-128 | 16,384 | h-means (EMA 0.4) T5 (torch cpu) | 37759.4 ± 2.777 | 15.4 ± 0.6 | 5.10 |
| SIFT-128 | 16,384 | h-means T3 (torch cpu) | 37931.4 ± 3.409 | 12.1 ± 1.1 | 3.10 |
| SIFT-128 | 16,384 | h-means T5 (torch cpu) | 37668.1 ± 1.538 | 19.0 ± 0.8 | 5.10 |
| SIFT-128 | 16,384 | lloyd@25 (torch cpu) | 38233.5 ± 5.856 | 100.5 ± 0.2 | 25.00 |
| SIFT-128 | 65,536 | e-means T5 (torch cpu) | 29717.9 ± 14.47 | 78.7 ± 8.1 | 5.00 |
| SIFT-128 | 65,536 | e-means official (faiss cpu) T5 | 29718.9 ± 16.68 | 304.0 ± 2.7 | 5.00 |
| SIFT-128 | 65,536 | h-means (EMA 0.8) T3 (torch cpu) | 29158 ± 4.047 | 45.4 ± 7.2 | 3.09 |
| SIFT-128 | 65,536 | h-means (EMA 0.8) T5 (torch cpu) | 28787.9 ± 3.321 | 66.4 ± 9.1 | 5.09 |
| SIFT-128 | 65,536 | h-means T3 (torch cpu) | 29191 ± 4.972 | 40.6 ± 2.7 | 3.09 |
| SIFT-128 | 65,536 | h-means T5 (torch cpu) | 28809 ± 5.263 | 61.0 ± 4.5 | 5.09 |