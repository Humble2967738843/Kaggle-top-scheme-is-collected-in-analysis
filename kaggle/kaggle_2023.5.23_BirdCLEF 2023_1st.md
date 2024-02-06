
| 竞赛平台 | kaggle                                                       |
| -------- | ------------------------------------------------------------ |
| 竞赛名称 | [BirdCLEF 2023]([https://www.kaggle.com/competitions/llm-detect-ai-generated-text/leaderboard](https://www.kaggle.com/competitions/birdclef-2023/overview)) |
| 竞赛排名 | 1st                                                          |
| 竞赛标签 | Multiclass Classification Audio                       |
| 方案地址 | https://www.kaggle.com/competitions/birdclef-2023/discussion/412808 https://www.kaggle.com/code/vladimirsydor/bird-clef-2023-inference-v1/notebook |

## 🤖🤖我的收获
大师傅似的

## 正确的数据就是您所需要的
## 数据，数据无处不在
让我们从 2023 年的训练数据开始

如果你看一下 `train_metadata["primary_label"].value_counts()`，你可能会注意到一些奇怪的最大幻数：

```python
barswa     500
wlwwar     500
thrnig1    500
eaywag1    500
comsan     500
          ... 
lotcor1      1
whctur2      1
whhsaw1      1
afpkin1      1
crefra2      1
Name: primary_label, Length: 264, dtype: int64
```
为什么某些物种的代表最多有 500 个？我不知道 100% 的答案，但我有一个强有力的假设 - XC API 中的错误。我不记得代码中的确切位置，但总体问题在于数据加载管道。它的工作原理如下：

1. 下载元文件 - json 文件。
2. 迭代元文件中的所有 url 并下载它们。

但是，如果一个物种有超过 500 个文件 - 在第一阶段，您将有几个 json 文件（一个 json 图元文件中的最大文件数 = 500），这时候我们就遇到了问题！在第二阶段，API 仅考虑每个物种的一个 json，并忽略接下来的 json，因此每个物种最多有 500 个文件。

注意：我不确定它是否在最新版本的 API 中得到修复，但我使用了前一年的提交，而且它就在那里。

从这个bug中，我们可以清楚地了解到使用固定的API可以极大地丰富我们的训练数据集。

其他比较无聊的事情
- 2023/2022/2021/2020比赛数据
- 2020年附加比赛数据
- 泽诺多
- 异曲

## 额外的训练数据

从2023/2022/2021/2020的比赛数据加上Xeno-Canto数据中，我只选择了带有今年主要标签的文件，并将它们添加到训练的最后阶段。

## 预训练数据集

当我在训练的最后阶段只使用2023年的训练数据时，对2022/2021/2020比赛数据进行预训练，分数提升了很多。但在添加额外的训练数据后，预训练在排行榜上停止了工作（尽管它仍然增加了本地验证）。上周，我决定重新开始预训练实验。这让我在公共排行榜上上升了一位，在私人排行榜上上升了两位 - 所以，Kagglers，不要忘记重新审视甚至被拒绝的假设:)

为什么以及何时起作用？与之前的预训练实验相比，我有：

- 不仅按 id 过滤掉 2023 个火车数据重复项，还按“author + Primary_label”过滤掉，如此处建议的那样
- 选取2023/2022/2021/2020年竞赛数据+2020年附加竞赛数据中存在的物种，且前提是该物种的代表数量超过10个。总共有822种。
- 添加了来自 Xeno Canto 的选定物种的附加文件。

## 泽诺多

我选择了 nocall 区域并将其用作背景增强。

## 无效的数据实验

- 对所有 Xeno Canto 数据进行大规模预训练。
- 2021 年第二名的背景噪音作为背景增强
- ESC50 作为背景增强
- 从附加数据中仅选择高质量样本 (>=32kHz)
- 也许我刚刚忘记了 200 多个实验中的一些其他想法

## 验证：像comAP一样软，不要像F1一样硬

最后！与音景数据相比，我们不必在完全不同的训练数据上选择阈值，提出超级复杂的方案或下降 19 位（就像我在 2021 年所做的那样）

我使用了与前几年的比赛几乎相同的验证方案：

- 简历分层 5 折
- 随着时间的推移，从所有样本的每 5 秒剪辑中获取最大概率
- 重要提示：对于带衬垫的 comAP，在折叠处取均值非常重要，而不是在折叠处进行！！！

当然，CV和LB的绝对数量是不同的：

最佳公共LB：0.84444（4个四:)）
最佳私人LB：0.76392
最佳简历：0.9083368282233681

但排名相关性非常好。 CV 改善 0.0 倍（或更多）导致 LB 改善。我的实验几乎拥有所有 CV 结果，因此我希望有时间发表一篇包含详细消融研究和 CV-LB 相关性研究的论文。

## 训练

我看了@philippsinger 的演示，了解了我一直以来的过度拟合程度。

由于时间和设备的限制，我选择了以下方案：

- 验证 CV 的假设并提交前 2-3 个折叠。
- 对于完整训练数据的集成重新训练，因此每个设置都有一个模型

培训详情：

- 50 Epochs
- Adam
- CosineAnnealing from 1e-4 (or 1e-3) to 1e-6
- Focal loss
- 64 BS
- 5 second chunk
- SUPER IMPORTANT: Class sampling weights

```python
sample_weights = (
    all_primary_labels.value_counts() / 
    all_primary_labels.value_counts().sum()
)  ** (-0.5)

```
- Same setups for pretrain and finetune

Stages:

- Pretrain - refer to Pretrained Dataset
- Tune only on scored species

## 模型
由于计算限制，我们无法使用深度学习的黄金法则：堆叠更多层！

所以我深入研究了推理优化技术：

- ONNX - 这对我来说效果很好。它稍微缩短了推理时间，并允许我减少推理笔记本中自定义依赖项的数量。
量化 - 我花了一个多星期的时间来尝试它，但不幸的是，我没有成功:(
- openvino - 我没有使用或尝试这个，我只是阅读了第二名的描述并烧毁了我的椅子

总的来说，我的最终提交是 3 个声音事件检测 (SED) 模型的集合，具有以下主干：

- eca_nfnet_l0（2 阶段训练；启动 LR 1e-3）
- convnext_small_fb_in22k_ft_in1k_384（2阶段训练；开始LR 1e-4）
- convnextv2_tiny_fcmae_ft_in22k_in1k_384（1阶段训练；开始LR 1e-4）

调整不同架构的起始学习率非常重要！

## 增强

我对增强选择非常挑剔，所以我的最终模型使用了下一个：

- Mixup : Simply OR Mixup with Prob = 0.5
- BackgroundNoise with Zenodo nocall
- RandomFiltering - a custom augmentation: in simple terms, it's a simplified random Equalizer
- Spec Aug:
  - Freq:
    - Max length: 10
    - Max lines: 3
    - Probability: 0.3
  - Time:
    - Max length: 20
    - Max lines: 3
    - Probability: 0.3

## 小推理技巧
- 使用温度平均值： `pred = (pred**2).mean(axis=0) ** 0.5`
- 使用注意力 SED 概率 * 0.75 + 最大时间概率 * 0.25

所有这些都带来了微小的改进，但这只是前三名的问题:)

## 其他产生碳足迹但没有提高我的 LB 分数的东西

本节还远未完成，但让我们添加一些我现在想到的内容：

- 2021 年第二名[方案](https://www.kaggle.com/competitions/birdclef-2023/discussion/412707)。我已经尝试过（就像我在 2022 年所做的那样），但不幸的是它对我不起作用
- 对整个 Xeno Canto 进行预训练
- 训练更大的块。如果我推断较小的块或相同长度的块，则会出现相同的结果
- 彩色噪声增强
- CQT 或叶
- 具体微调：较小的 LR、较小的 epoch 数量、冻结主干、主干和头部的不同 LR
- 注意力 SED 概率损失 + 最大时间概率损失
- 深度督导
- MixUp 的不同 alpha
- 变压器架构。例如 ECAPA TDNN

## 代码方案

```python
!nvidia-smi
```

    /bin/bash: nvidia-smi: command not found




```python
!pip install /kaggle/input/bird-clef-2023-addones/onnxruntime-1.14.0-cp37-cp37m-manylinux_2_27_x86_64.whl --no-deps
```

    Processing /kaggle/input/bird-clef-2023-addones/onnxruntime-1.14.0-cp37-cp37m-manylinux_2_27_x86_64.whl
    
    Installing collected packages: onnxruntime
    
    Successfully installed onnxruntime-1.14.0
    
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    
    [0m[33mWARNING: There was an error checking the latest version of pip.[0m[33m
    
    [0m


```python
!pip list | grep onnx
```

    onnx                                   1.13.1
    
    onnxruntime                            1.14.0
    
    [33mWARNING: There was an error checking the latest version of pip.[0m[33m
    
    [0m


```python
import sys
sys.path.append('/kaggle/input/bird-clef-2023-code/main_folder/main_folder/')
```


```python
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import os
import json
import IPython.display as ipd
import soundfile as sf
import torch
import h5py
import onnxruntime as ort

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from itertools import chain
from os.path import join as pjoin
from copy import deepcopy


from code_base.models import WaveCNNClasifier, WaveCNNAttenClasifier
from code_base.datasets import WaveDataset, WaveAllFileDataset
from code_base.utils.inference_utils import apply_avarage_weights_on_swa_path
from code_base.inefernce import BirdsInference
from code_base.utils import load_json, compose_submission_dataframe, groupby_np_array, stack_and_max_by_samples
from code_base.utils.metrics import padded_cmap_numpy
%matplotlib inline

```

    `speechbrain` was not imported
    `nnAudioSTFT` was not imported
    `noisereduce` was not imported


# Config


```python
EXP_NAME = "convnext_small_fb_in22k_ft_in1k_384__convnextv2_tiny_fcmae_ft_in22k_in1k_384__eca_nfnet_l0_noval_v32_075Clipwise025TimeMax_GausMean"
TRAIN_PERIOD = 5
print("Possible checkpoints:\n\n{}".format("\n".join(set([
    os.path.basename(el) for el in glob(f"/kaggle/input/bird-clef-2023-models/{EXP_NAME}/{EXP_NAME}/*/checkpoints/*.pt*") if "train" not in os.path.basename(el)
]))))
```

    Possible checkpoints:


​    
​    


```python
CONFIG = {
    # Main
    "run_validation": False,
    "run_test": True,
    # Inference Class
    "use_sigmoid": False,
    # Data config
#     "train_df_path":"/home/vova/data/exps/BirdCLEF_2023/birdclef_2023/train_metadata_extended.csv",
#     "split_path":"/home/vova/data/exps/BirdCLEF_2023/cv_split_2023_v1.npy",
    "folds":[0],
#     "train_data_root":"/home/vova/data/exps/BirdCLEF_2023/birdclef_2023/train_audio/",
#     "test_data_root": "/kaggle/input/bird-clef-2023-addones/fake_test_20/fake_test_20/*.ogg",
    "test_data_root": "/kaggle/input/birdclef-2023/test_soundscapes/*.ogg",
    "label_map_data_path":'/kaggle/input/bird-clef-2023-models/bird2int_2023.json',
#     "label_map_data_path":'/kaggle/input/bird-clef-2023-models/bird221int_202x.json',
#     "label_map_data_path": "/kaggle/input/bird-clef-2023-models/xc_birds_202x_only_scored.json",
#     "label_map_data_path": '/kaggle/input/bird-clef-2023-models/bird2id_xc_pretrain.json',
#     "label_map_data_path": "/kaggle/input/bird-clef-2023-models/bird2id.json",
#     "label_map_data_path":"/kaggle/input/bird-clef-2023-models/bird2id_v1.json",
    "lookback":None,
    "lookahead":None,
    "segment_len":5,
    "step": None,
    "late_normalize": True,
    # Model config
    "exp_name":EXP_NAME,
#     "model_class": WaveCNNAttenClasifier,
#     "model_config": dict(
#         backbone="convnext_tiny_in22ft1k",
#         mel_spec_paramms={
#             "sample_rate": 32000,
#             "n_mels": 128,
#             "f_min": 20,
#             "n_fft": 2048,
#             "hop_length": 512,
#             "normalized": True,
#         },
#         head_config={
#             "p": 0.5,
#             "num_class": 264,
#             "train_period": TRAIN_PERIOD,
#             "infer_period": TRAIN_PERIOD,
#         },
#         pretrained=False
#     ),
#     "chkp_name":"model.last.pth",
#     "swa_checkpoint": None,
#     "distributed_chkp": False,
}

if CONFIG.get("use_sed_mode", False):
    assert CONFIG["step"] is not None
else:
    assert CONFIG["step"] is None
    
if "folds" not in CONFIG:
    CONFIG["folds"] = list(range(CONFIG["n_folds"]))
```

# Data


```python
bird2id = load_json(CONFIG["label_map_data_path"])
```


```python
if CONFIG["run_validation"]:
    df = pd.read_csv(CONFIG["train_df_path"])
    split = np.load(CONFIG["split_path"], allow_pickle=True)
    val_df = [df.iloc[split[fold_id][1]].reset_index(drop=True) for fold_id in CONFIG["folds"]]
```


```python
if CONFIG["run_test"]:
    test_au_pathes = glob(CONFIG["test_data_root"])

    test_df = pd.DataFrame({
        "filename": test_au_pathes,
        "duration_s": [librosa.get_duration(filename=el) for el in test_au_pathes]
    })
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:6: FutureWarning: get_duration() keyword argument 'filename' has been renamed to 'path' in version 0.10.0.
    	This alias will be removed in version 1.0.


​    


```python
if CONFIG["run_validation"]:
    val_ds_conig = {
       "root": CONFIG["train_data_root"],
       "label_str2int_mapping_path": CONFIG["label_map_data_path"],
       "use_audio_cache": True,
       "n_cores": 64,
       "verbose": False,
       "segment_len": CONFIG["segment_len"],
       "lookback":CONFIG["lookback"],
       "lookahead":CONFIG["lookahead"],
       "sample_id": None,
       "late_normalize": CONFIG["late_normalize"],
       "step": CONFIG["step"],
       "validate_sr": 32_000
    }
if CONFIG["run_test"]:
    ds_config_test = {
       "root": "",
       "label_str2int_mapping_path": CONFIG["label_map_data_path"],
       "n_cores": 64,
       "use_audio_cache": True,
       "test_mode": True,
       "segment_len": CONFIG["segment_len"],
       "lookback":CONFIG["lookback"],
       "lookahead":CONFIG["lookahead"],
        "sample_id": None,
        "late_normalize": CONFIG["late_normalize"],
        "step": CONFIG["step"],
        "validate_sr": 32_000
    }
loader_config = {
    "batch_size": 4,
    "drop_last": False,
    "shuffle": False,
    "num_workers": 0,
}
```


```python
if CONFIG["run_test"]:
    ds_test = WaveAllFileDataset(df=test_df, **ds_config_test)
if CONFIG["run_validation"]:
    ds_val = [WaveAllFileDataset(df=df, **val_ds_conig) for df in val_df]
```

    secondary_labels is not found in df. Maybe test or nocall mode



```python
if CONFIG["run_validation"]:
    loader_val = [torch.utils.data.DataLoader(
        ds,
        **loader_config,
    )for ds in ds_val]
if CONFIG["run_test"]:
    loader_test = torch.utils.data.DataLoader(
        ds_test,
        **loader_config,
    )
```



# Model


```python
def create_model_and_upload_chkp(
    model_class,
    model_config,
    model_device,
    model_chkp,
    use_distributed=False,
    swa_checkpoint=None
):
    print(model_chkp)
    if "swa" in model_chkp:
        print("swa by {}".format(os.path.splitext(os.path.basename(model_chkp))[0]))
        t_chkp = apply_avarage_weights_on_swa_path(model_chkp, use_distributed=use_distributed, take_best=swa_checkpoint)
    else:
        print("vanilla model")
        t_chkp = torch.load(model_chkp, map_location="cpu")
        
    t_model = model_class(**model_config, device=model_device)
    t_model.load_state_dict(t_chkp)
    t_model.eval()
    return t_model
```
这段代码定义了一个函数`create_model_and_upload_chkp`，该函数的作用是创建模型并加载检查点文件。以下是对代码的分析：

1. 函数接受以下参数：
   - `model_class`: 模型类别，用于创建模型的类。
   - `model_config`: 包含模型配置信息的字典。
   - `model_device`: 指定模型所在的设备。
   - `model_chkp`: 模型的检查点文件路径。
   - `use_distributed`: 布尔值，表示是否使用分布式训练。默认为`False`。
   - `swa_checkpoint`: 如果模型使用了SWA（Stochastic Weight Averaging）方法，提供SWA检查点的路径。

2. 打印模型检查点路径：函数会打印出传入的`model_chkp`参数的值。

3. 判断是否是 SWA 模型：通过检查`model_chkp`是否包含 "swa" 来确定是否使用 SWA。如果是 SWA 模型，则打印相应的信息，并调用`apply_avarage_weights_on_swa_path`函数来应用 SWA 方法的平均权重。否则，打印 "vanilla model" 表示普通模型，并使用`torch.load`加载检查点文件。

4. 创建模型实例：使用`model_class`和`model_config`创建一个模型实例`t_model`，同时指定设备为`model_device`。

5. 加载模型参数：使用加载的检查点文件中的权重参数调用`t_model.load_state_dict(t_chkp)`加载模型参数。

6. 设置模型为评估模式：通过调用`t_model.eval()`将模型设置为评估模式。

7. 返回模型实例：返回创建并加载了检查点的模型实例`t_model`。

总体而言，这个函数的主要作用是根据给定的参数创建模型实例，并根据提供的检查点文件加载模型的权重参数。如果检查点文件名中包含 "swa"，则会应用 SWA 方法的平均权重。最终返回加载了权重参数的模型实例。

```python
# model = [create_model_and_upload_chkp(
#         model_class=CONFIG["model_class"],
#         model_config=CONFIG['model_config'],
#         model_device="cpu",
#         model_chkp=f"/kaggle/input/bird-clef-2023-models/{CONFIG['exp_name']}/{CONFIG['exp_name']}/fold_{m_i}/checkpoints/{CONFIG['chkp_name']}",
#         swa_checkpoint=CONFIG['swa_checkpoint'],
#         use_distributed=CONFIG['distributed_chkp']
# ) for m_i in CONFIG["folds"]]
```


```python
model = ort.InferenceSession(f"/kaggle/input/bird-clef-2023-models/{CONFIG['exp_name']}/{CONFIG['exp_name']}/checkpoints/model_simpl.onnx")
```

# Inference Class


```python
inference_class = BirdsInference(
    device="cpu",
    verbose_tqdm=True,
    use_sigmoid=CONFIG["use_sigmoid"],
#     model_output_key=CONFIG["model_output_key"],
)
```

# Val Predict


```python
if CONFIG["run_validation"]:
    val_tgts, val_preds, val_preds_long = inference_class.predict_val_loaders(
        nn_models=model,
        data_loaders=loader_val
    )
    print(
        f"Min prob {val_preds.min()}. Max prob {val_preds.max()}.\n"
        f"Padded CMAP: {padded_cmap_numpy(val_tgts, val_preds)}"
    )
```

# Test Pred


```python
if CONFIG["run_test"]:
    test_preds, test_preds_long, test_dfidx, test_end = inference_class.predict_test_loader(
        nn_models=model,
        data_loader=loader_test,
        is_onnx_model=True
    )
    test_pred_df = compose_submission_dataframe(
        probs=test_preds,
        dfidxs=test_dfidx,
        end_seconds=test_end,
        filenames=loader_test.dataset.df[loader_test.dataset.name_col].copy(),
        bird2id=bird2id,
    )
    sample_submission = pd.read_csv("/kaggle/input/birdclef-2023/sample_submission.csv")
    if test_pred_df.shape[1] > sample_submission.shape[1]:
        print("Shrinking columns")
        test_pred_df = test_pred_df[sample_submission.columns]
```

      0%|          | 0/30 [00:00<?, ?it/s]
    
    Loading /kaggle/input/birdclef-2023/test_soundscapes/soundscape_29201.ogg to audio cache


    100%|██████████| 30/30 [00:49<00:00,  1.65s/it]



```python
test_pred_df.iloc[:,1:].values.max(), test_pred_df.iloc[:,1:].values.min()
```




    (0.8837069272994995, 0.004084157757461071)




```python
plt.hist(test_pred_df.iloc[:,1:].values.max(axis=1), bins=30);
```


​    
![png](https://www.kaggleusercontent.com/kf/130572858/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..SHeM5U5kHKJQqmIz2MxOdw.Jv7dq963obw-94-6DgQTosF1St_osn-rY5qn8YajGzWqiFQrMBag_3vmZIgGFzymx2AsaReJHG9m2nrTbh1ittJ7krxrPYCDLp5730SMIJCBvsf0cO8JHCUz_Gueba1sG3IebiUzQx0axaJ7I7q3SFSGgiEy3IJnYfnsTnSf1u4BPR6QDfF8VuRQ4TxzRZoL_2afFUY4BD_dTrfhGqjbfGwzkmZCMAcEh-gEPIHJLt5-90du4FkuGXXWFSPQp9K-f4aD0IS85qBFbUbzfO8BOmlN_074yChJGL6riJWwH8zFPbJNQ-xq44ijyBdCIUsXzrtDTLMfRBCrZCuUe09GGPG_ePXl7Q_HHvFEO48Dd74z0dcHDu4bXpYNVB18coLIjCRuh6p56qHGhVhXIV0qaipskoMFG9fLkoTDkRx6cSbyM4PuCr3lDmYvnPxQSloz7Ym2JuxcwnIn_uJNVafMTAAQTRqofTerhDvsdU_xl0hiSavv-WaM6hqUDNbsF1LOI0N5HmLBWbZ-M1kx3yaovuGbP8Ki8VxDE_8T24J5CTYMYwZhzDcE3mqcsVJN74G8O8PlK6NtrIR3qx5d0TdOqPt5RxLAdhwk8ihzquRYO_nidDkFRSURjjwjI8FbsmxoLrI47vn7s7CS6c-3uTw2jnKw1c8_KCNklPAmvSRz7FKZQ5tftdb_fTBKgys1X5Bs.SopSi2l7sImJ59xNisRoiQ/__results___files/__results___26_0.png)
​    
这段代码是一个条件语句，根据`CONFIG["run_test"]`的值执行不同的操作。以下是对代码的分析：

1. `CONFIG`是一个配置字典，其中包含了程序的一些配置参数。

2. `CONFIG["run_test"]`是一个布尔值，表示是否运行测试。如果`CONFIG["run_test"]`为`True`，则执行以下操作；否则，跳过这段代码块。

3. 调用`inference_class.predict_test_loader`方法进行测试集推断（inference）：
   - `nn_models`: 使用的神经网络模型。
   - `data_loader`: 测试集的数据加载器。
   - `is_onnx_model`: 一个布尔值，表示模型是否为ONNX格式。

4. 调用`compose_submission_dataframe`方法：
   - `probs`: 测试集的预测概率。
   - `dfidxs`: 测试集索引。
   - `end_seconds`: 测试集中每个样本的结束时间。
   - `filenames`: 测试集文件名。
   - `bird2id`: 鸟类别到ID的映射。

5. 读取样本提交文件（sample_submission.csv）：
   - 使用`pd.read_csv`读取路径为"/kaggle/input/birdclef-2023/sample_submission.csv"的样本提交文件，将其存储在`sample_submission`变量中。

6. 检查测试预测数据框的列数是否大于样本提交文件的列数：
   - 如果`test_pred_df`的列数（特征数）大于`sample_submission`的列数，则打印 "Shrinking columns" 并将`test_pred_df`的列数截断为与`sample_submission`相同的列数。

总体而言，这段代码块的作用是在运行测试时，进行测试集的推断（inference），生成提交的数据框，并在必要时截断列以符合样本提交文件的格式。

# Map Predictions


```python
if CONFIG.get("check_inf_class", False):
    val_ds_check = WaveAllFileDataset(df=val_df[0], **{
           "root": CONFIG["train_data_root"],
           "label_str2int_mapping_path": CONFIG["label_map_data_path"],
           "n_cores": 64,
           "use_audio_cache": True,
           "test_mode": True,
           "segment_len": CONFIG["segment_len"],
           "lookback":CONFIG["lookback"],
           "lookahead":CONFIG["lookahead"],
            "sample_id": None,
            "late_normalize": CONFIG["late_normalize"],
            "step": CONFIG["step"],
        }
    )
    val_loader_check = torch.utils.data.DataLoader(
        val_ds_check,
        **loader_config
    )
    
    test_preds_check, test_preds_long_check, test_dfidx_check, test_end_check = inference_class.predict_test_loader(
        nn_models=model,
        data_loader=val_loader_check
    )
    test_preds_check_grouped = groupby_np_array(
        groupby_f=test_dfidx_check,
        array_to_group=test_preds_check,
        apply_f=stack_and_max_by_samples,
    )
    print(np.allclose(
        test_preds_check_grouped,
        val_preds
    ))
```

# Boost Probs


```python
# CLASSES = list(set(test_pred_df.columns[1:]))
# print(len(CLASSES))
```


```python
# def connected_region_indices(bool_array):
#     connected_regions = []
#     region = []

#     for i, value in enumerate(bool_array):
#         if value:
#             region.append(i)
#         elif region:
#             connected_regions.append(region)
#             region = []

#     if region:
#         connected_regions.append(region)

#     return connected_regions

# def modify_probabilities(group, lower_prob=0.5, max_prob=0.75, min_seq_len=2):
#     class_cols = ['shesta1', 'colsun2', 'amesun2', 'bcbeat1', 'marsun2']
    
#     mask_low = (group[CLASSES] > lower_prob).values
#     mask_high = (group[CLASSES] > max_prob).values
    
#     # At least 2 chunks AND exceeds max_prob
#     classes_to_boost = np.where((mask_low.sum(axis=0) >= min_seq_len) & mask_high.any(axis=0))[0]
#     if len(classes_to_boost) > 0:
#         for cls in classes_to_boost:
#             new_probs = group[CLASSES[cls]].values.copy()
#             connected_regions = connected_region_indices(mask_low[:,cls])
#             for region in connected_regions:
#                 if len(region) >= min_seq_len:
#                     max_region_prob = group[CLASSES[cls]].iloc[region].max()
#                     if max_region_prob > max_prob:
#                         # print(f"Boosting: {CLASSES[cls]}")
#                         new_probs[region] = max_region_prob
#             group[CLASSES[cls]] = new_probs
                
#     return group
```


```python
# test_pred_df["id"] = test_pred_df["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
# test_pred_df["sec"] = test_pred_df["row_id"].apply(lambda x: int(x.split("_")[-1]))

# test_pred_df = test_pred_df.sort_values(["id", "sec"]).reset_index(drop=True)

# test_pred_df = test_pred_df.groupby("id").apply(modify_probabilities).reset_index(drop=True)

# test_pred_df = test_pred_df.drop(columns=["id", "sec"])
```


```python
# np.where((test_pred_df_new[CLASSES].values != test_pred_df[CLASSES].values).sum(axis=0))
# test_pred_df_new.loc[(test_pred_df_new[CLASSES[185]] - test_pred_df[CLASSES[185]]) > 0, CLASSES[185]]
# test_pred_df.loc[(test_pred_df_new[CLASSES[185]] - test_pred_df[CLASSES[185]]) > 0, CLASSES[185]]
```


```python
# def boost_pred_prob(
#     input_df,
#     class_columns,
#     select_prob=0.8,
#     boost_prob=0.2
# ):
#     result_df = input_df.copy()
#     for sample_id in tqdm(set(input_df["id"])):
#         sample_id_pred_df = result_df.loc[result_df["id"] == sample_id, class_columns]
#         boost_classes = class_columns[sample_id_pred_df.values.max(axis=0) > select_prob]
#         boost_mask = sample_id_pred_df[boost_classes].values < select_prob - boost_prob
#         result_df.loc[result_df["id"] == sample_id, boost_classes] = (
#             (boost_mask.astype(np.float32) * (sample_id_pred_df[boost_classes].values + boost_prob)) +
#             ((~boost_mask).astype(np.float32) * sample_id_pred_df[boost_classes].values)
#         )
#     return result_df

# test_pred_df["id"] = test_pred_df["row_id"].apply(lambda x: "_".join(x.split("_")[:-1]))
# test_pred_df["sec"] = test_pred_df["row_id"].apply(lambda x: int(x.split("_")[-1]))

# test_pred_df = boost_pred_prob(
#     input_df=test_pred_df,
#     class_columns=test_pred_df.columns[1:-2]
# )
# test_pred_df = test_pred_df.drop(columns=["id", "sec"])
```


```python
# test_pred_df.iloc[:,1:].values.max(), test_pred_df.iloc[:,1:].values.min()
```


```python
# plt.hist(test_pred_df.iloc[:,1:].values.max(axis=1), bins=30);
```

# Save Prediction


```python
sample_submission = pd.read_csv("/kaggle/input/birdclef-2023/sample_submission.csv")
assert set(sample_submission.columns) == set(test_pred_df.columns)
test_pred_df = test_pred_df[sample_submission.columns]
test_pred_df.to_csv("submission.csv", index=False)
```


```python

```