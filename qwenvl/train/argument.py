import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    teacher_model_name_or_path: Optional[str] = field(default=None)
    train_teacher: bool = field(default=False)

    # 回归模型归一化参数
    waypoint_stats_path: Optional[str] = field(default=None,
        metadata={"help": "Path to waypoint stats JSON file for normalization"})
    num_waypoints: int = field(
        default=6,
        metadata={"help": "Number of future waypoints predicted by the trajectory head"}
    )
    waypoint_dim: int = field(
        default=3,
        metadata={"help": "Dimension of each waypoint, e.g. 3 for (x, y, z)"}
    )

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    eval_dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 324)
    min_pixels: int = field(default=28 * 28 * 324)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

    img_dir: str = field(default="./data")
    inference: bool = field(default=False)

    random: bool = field(default=True)
    kd_data: bool = field(default=False)
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    deepspeed_config: str = field(default="./EvoDriveVLA/config/deepspeed_test.json")
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=16)
    remove_unused_columns: bool = field(default=False)

    report_to: Optional[List[str]] = field(default_factory=lambda: ["tensorboard"])
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: int = field(default=10)

    attn_implementation: str = field(default="flash_attention_2")

    encoder_kd: bool = field(default=False)
    encoder_loss_weight: float = field(default=0.5)

    llm_kd: bool = field(default=False)

    logits_loss: bool = field(default=False)
    logits_loss_weight: float = field(default=0.5)
    logits_loss_temperature: float = field(default=1.0)

    hs_loss: bool = field(default=False)
    hs_loss_weight: float = field(default=0.5)

    # LearnableQ 输出模式
    use_lm_head: bool = field(
        default=False,
        metadata={"help": "True → lm_head + CE loss（18 query tokens，每个预测一个坐标的 vocab token）；"
                          "False → regression head + Smooth L1（6 query tokens，默认）"}
    )

@dataclass
class EvalArguments:
    eval_save_path: str = field(default='')    
    max_new_tokens: int = field(default=1024)
