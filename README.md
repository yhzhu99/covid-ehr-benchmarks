# covid_emr

## Prediction Tasks

- [x] (Early) Mortality Outcome Prediction
- [x] Leng-of-stay prediction

## Model Zoo

### Machine Learning Models

- [x] Random forest (RF)
- [x] Decision tree (DT)
- [x] Gradient Boosting Decision Tree (GBDT)
- [x] XGBoost
- [x] CatBoost

### Deep Learning Models

- [x] Multi-layer perceptron (MLP)
- [x] Recurrent neural network (RNN)
- [x] Long-short term memory network (LSTM)
- [x] Gated recurrent units (GRU)
- [x] Temporal convolutional networks
- [x] Transformer

### EHR Predictive Models

- [x] RETAIN
- [x] StageNet
- [x] Dr. Agent
- [x] AdaCare
- [x] ConCare
- [x] GRASP

## Requirements

- Python 3.7+
- PyTorch 1.10+
- Cuda 10.2+ (If you plan to use GPU)

Note:

- Most models can be run quickly on CPU.
- You are required to have a GPU with 12GB memory to run ConCare model on CDSL dataset.
- TCN model may run much faster on CPU.

## Usage

- Install requirements.

    ```bash
    pip install -r requirements.txt [-i https://pypi.tuna.tsinghua.edu.cn/simple] # [xxx] is optional
    ```

- Download TJH dataset from [An interpretable mortality prediction model for COVID-19 patients](https://www.nature.com/articles/s42256-020-0180-7), unzip and put it in `datasets/tongji/raw_data/` folder.
- Run preprocessing notebook. (You can skip this step if you have already done this in the later training process)
- (The CDSL dataset is also the same process.) You need to apply for the CDSL dataset if necessary. [Covid Data Save Lives Dataset](https://www.hmhospitales.com/coronavirus/covid-data-save-lives/english-version)
- Run following commands to train models.

    ```bash
    python main.py --cfg configs/xxx.yaml [--cuda CUDA_NUM] [--db]
    # Note:
    # 1) If you plan to use CUDA, use --cuda 0/1/2/...
    # 2) If you have configured database settings, you can use --db to upload performance after training to the database.
    ```

## Data Format

The shape and meaning of the tensor fed to the models are as follows:

- `x.pkl`: (N, T, D) tensor, where N is the number of patients, T is the number of time steps, and D is the number of features. At $D$ dimention, the first $x$ features are demographic features, the next $y$ features are lab test features, where $x + y = D$
- `y.pkl`: (N, T, 2) tensor, where the 2 values are [outcome, length-of-stay] for each time step.
- `visits_length.pkl`: (N, ) tensor, where the value is the number of visits for each patient.
- `missing_mask.pkl`: same shape as `x.pkl`, tell whether features are imputed. `1`: existing, `0`: missing.

Pre-processed data are stored in `datasets/tongji/processed_data/` folder.

## Configs

Below is the configurations after hyperparameter selection.

<details>

<summary>ML models</summary>

```bash
hm_los_catboost_kf10_md6_iter150_lr0.1_test
hm_los_decision_tree_kf10_md10_test
hm_los_gbdt_kf10_lr0.1_ss0.8_ne100_test
hm_los_random_forest_kf10_md10_mss2_ne100_test
hm_los_xgboost_kf10_lr0.01_md5_cw3_test
hm_outcome_catboost_kf10_md3_iter150_lr0.1_test
hm_outcome_decision_tree_kf10_md10_test
hm_outcome_gbdt_kf10_lr0.1_ss0.6_ne100_test
hm_outcome_random_forest_kf10_md20_mss10_ne100_test
hm_outcome_xgboost_kf10_lr0.1_md7_cw3_test
tj_los_catboost_kf10_md3_iter150_lr0.1_test
tj_los_decision_tree_kf10_md10_test
tj_los_gbdt_kf10_lr0.1_ss0.8_ne100_test
tj_los_random_forest_kf10_md20_mss5_ne100_test
tj_los_xgboost_kf10_lr0.01_md5_cw1_test
tj_outcome_catboost_kf10_md3_iter150_lr0.1_test
tj_outcome_decision_tree_kf10_md10_test
tj_outcome_gbdt_kf10_lr0.1_ss0.6_ne100_test
tj_outcome_random_forest_kf10_md20_mss2_ne10_test
tj_outcome_xgboost_kf10_lr0.1_md5_cw5_test
```

</details>

<details>
<summary>DL/EHR models</summary>

```bash
tj_outcome_grasp_ep100_kf10_bs64_hid64
tj_los_grasp_ep100_kf10_bs64_hid128
tj_outcome_concare_ep100_kf10_bs64_hid128
tj_los_concare_ep100_kf10_bs64_hid128
tj_outcome_agent_ep100_kf10_bs64_hid128
tj_los_agent_ep100_kf10_bs64_hid64
tj_outcome_adacare_ep100_kf10_bs64_hid64
tj_los_adacare_ep100_kf10_bs64_hid64
tj_outcome_transformer_ep100_kf10_bs64_hid128
tj_los_transformer_ep100_kf10_bs64_hid64
tj_outcome_tcn_ep100_kf10_bs64_hid128
tj_los_tcn_ep100_kf10_bs64_hid128
tj_outcome_stagenet_ep100_kf10_bs64_hid64
tj_los_stagenet_ep100_kf10_bs64_hid64
tj_outcome_rnn_ep100_kf10_bs64_hid64
tj_los_rnn_ep100_kf10_bs64_hid128
tj_outcome_retain_ep100_kf10_bs64_hid128
tj_los_retain_ep100_kf10_bs64_hid128
tj_outcome_mlp_ep100_kf10_bs64_hid64
tj_los_mlp_ep100_kf10_bs64_hid128
tj_outcome_lstm_ep100_kf10_bs64_hid64
tj_los_lstm_ep100_kf10_bs64_hid128
tj_outcome_gru_ep100_kf10_bs64_hid64
tj_los_gru_ep100_kf10_bs64_hid128
tj_multitask_rnn_ep100_kf10_bs64_hid64
tj_multitask_lstm_ep100_kf10_bs64_hid128
tj_multitask_gru_ep100_kf10_bs64_hid128
tj_multitask_transformer_ep100_kf10_bs64_hid128
tj_multitask_tcn_ep100_kf10_bs64_hid64
tj_multitask_mlp_ep100_kf10_bs64_hid128
tj_multitask_adacare_ep100_kf10_bs64_hid128
tj_multitask_agent_ep100_kf10_bs64_hid64
tj_multitask_concare_ep100_kf10_bs64_hid128
tj_multitask_stagenet_ep100_kf10_bs64_hid64
tj_multitask_grasp_ep100_kf10_bs64_hid128
tj_multitask_retain_ep100_kf10_bs64_hid64
hm_outcome_mlp_ep100_kf10_bs64_hid64
hm_los_mlp_ep100_kf10_bs64_hid128
hm_outcome_lstm_ep100_kf10_bs64_hid64
hm_los_lstm_ep100_kf10_bs64_hid128
hm_outcome_gru_ep100_kf10_bs64_hid64
hm_los_gru_ep100_kf10_bs64_hid128
hm_outcome_grasp_ep100_kf10_bs64_hid64
hm_los_grasp_ep100_kf10_bs64_hid64
hm_outcome_concare_ep100_kf10_bs64_hid128
hm_los_concare_ep100_kf10_bs64_hid64
hm_outcome_agent_ep100_kf10_bs64_hid128
hm_los_agent_ep100_kf10_bs64_hid64
hm_outcome_adacare_ep100_kf10_bs64_hid64
hm_los_adacare_ep100_kf10_bs64_hid128
hm_outcome_transformer_ep100_kf10_bs64_hid128
hm_los_transformer_ep100_kf10_bs64_hid128
hm_outcome_tcn_ep100_kf10_bs64_hid128
hm_los_tcn_ep100_kf10_bs64_hid128
hm_outcome_stagenet_ep100_kf10_bs64_hid64
hm_los_stagenet_ep100_kf10_bs64_hid64
hm_outcome_rnn_ep100_kf10_bs64_hid64
hm_los_rnn_ep100_kf10_bs64_hid128
hm_outcome_retain_ep100_kf10_bs64_hid128
hm_los_retain_ep100_kf10_bs64_hid128
hm_multitask_rnn_ep100_kf10_bs512_hid128
hm_multitask_lstm_ep100_kf10_bs512_hid64
hm_multitask_gru_ep100_kf10_bs512_hid128
hm_multitask_transformer_ep100_kf10_bs512_hid64
hm_multitask_tcn_ep100_kf10_bs512_hid128
hm_multitask_mlp_ep100_kf10_bs512_hid128
hm_multitask_adacare_ep100_kf10_bs512_hid128
hm_multitask_agent_ep100_kf10_bs512_hid128
hm_multitask_concare_ep100_kf10_bs64_hid128
hm_multitask_stagenet_ep100_kf10_bs512_hid128
hm_multitask_grasp_ep100_kf10_bs512_hid64
hm_multitask_retain_ep100_kf10_bs512_hid128
```
</details>

<details>
<summary>Two stage configs</summary>

```bash
tj_twostage_adacare_kf10.yaml
tj_twostage_agent_kf10.yaml
tj_twostage_concare_kf10.yaml
tj_twostage_gru_kf10.yaml
tj_twostage_lstm_kf10.yaml
tj_twostage_mlp_kf10.yaml
tj_twostage_retain_kf10.yaml
tj_twostage_rnn_kf10.yaml
tj_twostage_stagenet_kf10.yaml
tj_twostage_tcn_kf10.yaml
tj_twostage_transformer_kf10.yaml
tj_twostage_grasp_kf10.yaml
hm_twostage_adacare_kf10.yaml
hm_twostage_agent_kf10.yaml
hm_twostage_concare_kf10.yaml
hm_twostage_gru_kf10.yaml
hm_twostage_lstm_kf10.yaml
hm_twostage_mlp_kf10.yaml
hm_twostage_retain_kf10.yaml
hm_twostage_rnn_kf10.yaml
hm_twostage_stagenet_kf10.yaml
hm_twostage_tcn_kf10.yaml
hm_twostage_transformer_kf10.yaml
hm_twostage_grasp_kf10.yaml
```

</details>
