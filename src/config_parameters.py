from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.model.model_infrastructure import validate_scan_layout


class Section(BaseModel):
    """Base for configuration sections: a misspelt setting is an error, not a silent default."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(Section):
    """Validated architecture dimensions and representation settings."""
    esm_layer: int = Field(default=33, ge=0, le=33)
    convolution_scan_sizes: list[PositiveInt] = Field(
        default=[1, 2, 3, 4], description="Kernel size of each convolution stage."
    )
    convolution_scan_dimensions: list[PositiveInt] = Field(
        default=[256, 64, 16, 4], description="Output feature size of each convolution stage."
    )

    @model_validator(mode="after")
    def _check_scan_layout(self) -> "ModelConfig":
        validate_scan_layout(tuple(self.convolution_scan_sizes), tuple(self.convolution_scan_dimensions))
        return self


class TrainingConfig(Section):
    """Validated optimization, batching, and checkpoint-retention settings."""
    batch_size: PositiveInt = Field(default=4)
    num_epochs: PositiveInt = Field(
        default=5,
        description="ddG training epochs. Measured on one split's testing set at "
                    "learning_rate 1e-4, both correlations peak at 5 epochs and are flat or "
                    "slightly worse by 10; the training loss keeps falling, so later epochs buy "
                    "memorisation rather than generalisation.",
    )
    pka_num_epochs: PositiveInt = Field(default=100)
    learning_rate: PositiveFloat = Field(
        default=1e-4,
        description="Learning rate for everything except the pKa pipeline. At batch_size 16 a "
                    "run is only ~1,500 optimiser steps, and 1e-5 undertrains: measured on a "
                    "split's testing set, 1e-4 reaches 0.50 Pearson against 0.43 at 1e-5.",
    )
    pka_learning_rate: PositiveFloat = Field(
        default=1e-4,
        description="Learning rate for the pKa pipeline during PKAD pretraining. Measured on PKAD "
                    "chains excluded from pretraining, 1e-4 reaches 0.49 pKa units of mean absolute "
                    "error against 0.60 at 5e-6, which undertrains in the configured 20 epochs.",
    )
    pka_finetune_learning_rate: float = Field(
        default=1e-7, ge=0.0,
        description="Learning rate for the pKa pipeline during ddG training. An absolute value, "
                    "not a fraction of pka_learning_rate. At 1e-7 the pKa head moves by well "
                    "under a percent over a run, which is deliberate: raising it to 1e-6 or 1e-5 "
                    "was measured to degrade the PKAD fit four- to thirteen-fold for no "
                    "improvement in ddG. Set to 0 to freeze the head outright.",
    )
    weight_decay: float = Field(default=1e-5, ge=0.0)
    ddg_loss_delta: PositiveFloat = Field(default=1.0)
    random_seed: int = Field(default=42)
    evaluation_splits: int = Field(
        default=10, ge=2,
        description="Number of training splits --prepare divides the curated data into. Each "
                    "split holds out one part and trains on the rest, so ten splits hold out a "
                    "tenth each.",
    )
    evaluation_states: int = Field(
        default=3, ge=1,
        description="Number of initial states --train_for_evaluation trains per training split. "
                    "Each state fixes the starting weights, so state x begins from the same "
                    "weights in every split and the spread across states measures run-to-run "
                    "variation rather than data variation.",
    )
    max_files_kept: PositiveInt = Field(default=3)

class HardwareConfig(Section):
    """Validated device selection and data-loader worker settings."""
    device: Literal["auto", "cpu", "cuda"] = Field(default="auto")
    num_workers: int = Field(default=0, ge=0)

class DataProcessingConfig(Section):
    """Validated source-curation, embedding, and experiment processing limits."""
    max_sequence_length: PositiveInt = Field(default=2000)
    embedding_chunk_size: PositiveInt = Field(default=500)
    experiment_chunk_size: PositiveInt = Field(default=500)
    esm_batch_size: PositiveInt = Field(default=16)
    esm_attention_budget: PositiveInt = Field(
        default=4_000_000,
        description="Largest batch_size x padded_length^2 allowed in one ESM batch. ESM-2's "
                    "attention allocates batch x heads x length^2 values, so a fixed batch size "
                    "that is comfortable for short sequences runs a GPU out of memory on long "
                    "ones. At the default, batches of short sequences still reach esm_batch_size "
                    "while a 1800-residue sequence is embedded on its own. Lower it if the GPU "
                    "still runs out of memory.",
    )
    mad_threshold: PositiveFloat = Field(default=1.5)
    sem_tolerance: float = Field(default=0.25, ge=0.0)
    min_seq_id: float = Field(default=0.3, ge=0.0, le=1.0)
    coverage: float = Field(default=0.8, gt=0.0, le=1.0)
    cov_mode: int = Field(default=0, ge=0, le=2)

class RemoteConfig(Section):
    """Settings for running --find_stable_mutant on a Kaggle GPU (``--remote kaggle``)."""
    kaggle_username: str = Field(default="", description="Your Kaggle username; required for --remote kaggle.")
    kaggle_slug: str = Field(
        default="ddg-predict", pattern=r"^[a-z0-9][a-z0-9-]{4,40}$",
        description="Prefix for the private Kaggle dataset and script this toolkit creates.",
    )
    kaggle_machine_shape: str = Field(default="NvidiaTeslaT4", description="Kaggle accelerator to request.")
    kaggle_session_hours: float = Field(
        default=11.0, gt=0.0, le=11.5,
        description="Each Kaggle session stops itself safely after this long (Kaggle's limit is 12 hours).",
    )
    kaggle_poll_minutes: float = Field(default=5.0, ge=1.0, description="How often to check on the Kaggle run.")


class SetConfig(BaseSettings):
    """Top-level validated settings loaded from JSON, environment, or CLI."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    data: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_prefix="DDG_",
        env_nested_delimiter="__",
    )
