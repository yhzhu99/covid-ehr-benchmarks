# Dataset

## Supported datasets

- [x] Tongji Hospital
- [x] HM Hospital
- [ ] Synthea dataset

## Folder structure

```shell
dataset/
    README.md
    tongji/
        preprocess.ipynb
        raw_data/
            ...
        processed_data/
            train_x_demographic.pkl
            train_x_labtest.pkl
            train_y_{task}.pkl
    hm/
        ...
```

## Tongji Hospital Dataset

> Refer: [An interpretable mortality prediction model for COVID-19 patients](https://www.nature.com/articles/s42256-020-0180-7)

### How to Access Data

Download Link: [Supplementary Data 1 Training and external test datasets](https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-020-0180-7/MediaObjects/42256_2020_180_MOESM3_ESM.zip) (refer supplementary information of the paper above)
