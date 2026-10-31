# Human Activity Recognition on UCI HAR: 1D-CNN vs BiLSTM vs Transformer

ICT 4442 Deep Learning mini project. Team: Anuran Baruah (1D-CNN), Syed Ayaan Shere (BiLSTM), Raghav Nagpal (Transformer).

## Results (test set, 9 unseen subjects, mean ± std over seeds 42, 7, 123)

| Model | Accuracy % | Macro F1 % | Params | CPU ms/window (batch 1) |
|---|---|---|---|---|
| 1D-CNN | 93.15 ± 0.63 | 93.32 ± 0.51 | 127,494 | 0.29 |
| BiLSTM + attention | 90.71 ± 1.20 | 90.90 ± 1.11 | 146,823 | 1.70 |
| Transformer encoder | 88.01 ± 1.58 | 87.93 ± 1.65 | 101,574 | 1.28 |

## Layout

- `code/data.py`: shared preprocessing and subject-wise split (train 5,905 / val 1,447 / test 2,947)
- `code/models.py`: CNN1D, BiLSTMAttn, TransformerHAR
- `code/train.py`: shared training and evaluation
- `code/eval_extra.py`: latency, per-subject accuracy, mean/std summary
- `code/plots_final.py`: figures
- `results/`: per-run JSONs, checkpoints, summary.json, figures
- `report/`: final report (Part C)

## Reproduce

1. Download the UCI HAR dataset ("UCI HAR Dataset" folder) from the UCI ML Repository. It is not included in this repo.
2. `pip install -r requirements.txt`
3. From `code/`, for each model in `cnn bilstm transformer` and seed in `42 7 123`:
   `python train.py --model <model> --data "<path>/UCI HAR Dataset" --out ../results --seed <seed>`
4. `python eval_extra.py --data "<path>/UCI HAR Dataset" --res ../results`
5. `python plots_final.py ../results`

Use of AI tools: an LLM (Claude) was used to write the code, run the experiments and draft the report (see the report's AI-use statement).
