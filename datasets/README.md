# Dataset

## Supported datasets

- [x] Tongji Hospital (aka. TJH dataset)
- [x] HM Hospital (aka. CDSL dataset)

## Folder structure

```shell
datasets/
    tongji/
        preprocess.py
        raw_data/
            ...
        processed_data/
            fold_{i=0,1,2,...}/
                train/
                    x.pkl
                    y.pkl
                    statistics.csv
                val/
                    x.pkl
                    y.pkl
                test/
                    x.pkl
                    y.pkl
    hm/
        preprocess.py
        ...
```

We also provide a Jupyter format pre-process script (`preprocess.ipynb`)

## TJH Dataset

> Refer: [An interpretable mortality prediction model for COVID-19 patients](https://www.nature.com/articles/s42256-020-0180-7)

### How to Access Data

Download Link: [Supplementary Data 1 Training and external test datasets](https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-020-0180-7/MediaObjects/42256_2020_180_MOESM3_ESM.zip) (refer supplementary information of the paper above)

Or you can download raw data (Excel format) and processed data (Python PKL format) in GitHub Releases Assets.

## CDSL Dataset

You need to apply for the CDSL dataset if you need. [Link: Covid Data Save Lives Dataset](https://www.hmhospitales.com/coronavirus/covid-data-save-lives/english-version)
