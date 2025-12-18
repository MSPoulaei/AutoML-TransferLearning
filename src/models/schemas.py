# src/models/schemas.py

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class MetricType(str, Enum):
    """Supported metric types."""

    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC_ROC = "auc_roc"
    TOP_K_ACCURACY = "top_k_accuracy"


class DatasetDomain(str, Enum):
    """Dataset domain categories."""

    NATURAL = "natural"
    MEDICAL = "medical"
    SATELLITE = "satellite"
    DOCUMENT = "document"
    FINE_GRAINED = "fine_grained"
    INDUSTRIAL = "industrial"
    ARTISTIC = "artistic"
    OTHER = "other"


class FinetuningType(str, Enum):
    """Available fine-tuning strategies."""

    FULL_FINETUNING = "full_finetuning"
    HEAD_ONLY = "head_only"
    GRADUAL_UNFREEZING = "gradual_unfreezing"
    DISCRIMINATIVE_LR = "discriminative_lr"


class DatasetInfo(BaseModel):
    """Information about the dataset without naming it."""

    num_classes: int = Field(..., ge=2, description="Number of classification classes")
    num_samples: int = Field(
        ..., ge=10, description="Approximate number of training samples"
    )
    image_size: tuple[int, int] = Field(..., description="Expected image size (H, W)")
    num_channels: int = Field(
        default=3, ge=1, le=4, description="Number of image channels"
    )
    domain: DatasetDomain = Field(..., description="Domain of the images")
    domain_description: str = Field(
        ..., description="Detailed description of image content"
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Optional dataset name for auto-loading (e.g., 'cifar10', 'mnist')",
    )
    class_balance: str = Field(
        default="balanced",
        description="Class distribution: balanced, slightly_imbalanced, highly_imbalanced",
    )
    data_quality: str = Field(
        default="high", description="Data quality: high, medium, low"
    )
    primary_metric: MetricType = Field(
        default=MetricType.ACCURACY, description="Primary metric to optimize"
    )
    secondary_metrics: list[MetricType] = Field(
        default_factory=list, description="Additional metrics to track"
    )
    has_augmentation: bool = Field(
        default=True, description="Whether data augmentation is available"
    )
    validation_split: float = Field(
        default=0.2, ge=0.05, le=0.5, description="Validation split ratio"
    )

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v):
        if v[0] < 28 or v[1] < 28:
            raise ValueError("Image size must be at least 28x28")
        return v


class BackboneConfig(BaseModel):
    """Configuration for a backbone model."""

    family: str = Field(..., description="Backbone family (resnet, efficientnet, etc.)")
    variant: str = Field(
        ..., description="Specific variant (resnet50, efficientnet_b0, etc.)"
    )
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    input_size: tuple[int, int] = Field(
        default=(224, 224), description="Expected input size"
    )
    estimated_memory_mb: float = Field(
        default=1024, description="Estimated memory usage in MB"
    )
    num_features: int = Field(default=2048, description="Number of output features")

    @property
    def full_name(self) -> str:
        return f"{self.family}/{self.variant}"


class FinetuningStrategy(BaseModel):
    """Configuration for fine-tuning strategy."""

    strategy_type: FinetuningType = Field(..., description="Type of fine-tuning")
    learning_rate: float = Field(default=1e-3, gt=0, description="Base learning rate")
    weight_decay: float = Field(default=1e-4, ge=0, description="Weight decay")
    freeze_bn: bool = Field(
        default=False, description="Freeze batch normalization layers"
    )
    memory_multiplier: float = Field(
        default=2.0, description="Memory overhead multiplier"
    )

    # Strategy-specific parameters
    gradual_unfreeze_epochs: Optional[int] = Field(
        default=None,
        description="Epochs between unfreezing stages (for gradual_unfreezing)",
    )
    layer_lr_decay: Optional[float] = Field(
        default=None,
        description="Learning rate decay per layer (for discriminative_lr)",
    )


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    backbone: BackboneConfig
    strategy: FinetuningStrategy

    # Training hyperparameters
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=32, ge=1, le=512)
    optimizer: str = Field(default="adamw")
    scheduler: str = Field(default="cosine")
    warmup_epochs: int = Field(default=5, ge=0)

    # Regularization
    dropout: float = Field(default=0.2, ge=0, le=0.9)
    label_smoothing: float = Field(default=0.1, ge=0, le=0.3)
    mixup_alpha: float = Field(default=0.0, ge=0)
    cutmix_alpha: float = Field(default=0.0, ge=0)

    # Early stopping
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=10, ge=1)

    # Computed
    estimated_total_memory_mb: float = Field(default=0)

    def calculate_memory(self, dataset_info: DatasetInfo) -> float:
        """Estimate total memory usage."""
        backbone_mem = self.backbone.estimated_memory_mb
        strategy_mult = self.strategy.memory_multiplier

        # Batch memory estimate
        batch_mem = (
            self.batch_size
            * dataset_info.num_channels
            * dataset_info.image_size[0]
            * dataset_info.image_size[1]
            * 4  # float32
        ) / (
            1024 * 1024
        )  # Convert to MB

        total = backbone_mem * strategy_mult + batch_mem * 2  # Forward + backward
        self.estimated_total_memory_mb = total
        return total


class TrainingResult(BaseModel):
    """Results from a training run."""

    config: TrainingConfig

    # Metrics
    train_loss: float
    val_loss: float
    primary_metric_value: float
    primary_metric_name: MetricType
    secondary_metrics: dict[str, float] = Field(default_factory=dict)

    # Training info
    epochs_trained: int
    best_epoch: int
    training_time_seconds: float
    stopped_early: bool = False

    # History
    train_loss_history: list[float] = Field(default_factory=list)
    val_loss_history: list[float] = Field(default_factory=list)
    metric_history: list[float] = Field(default_factory=list)

    # Model info
    model_path: Optional[str] = None
    model_size_mb: float = 0


class AnalyzerRecommendation(BaseModel):
    """Recommendation from the Analyzer Agent."""

    iteration: int
    training_config: TrainingConfig

    # Reasoning
    reasoning: str = Field(
        ..., description="Explanation of why this configuration was chosen"
    )
    expected_performance: float = Field(
        ..., ge=0, le=1, description="Expected metric value"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in this recommendation"
    )

    # Alternative suggestions
    alternatives: list[dict[str, Any]] = Field(
        default_factory=list, description="Alternative configurations to try"
    )

    # Memory check
    memory_check_passed: bool = True
    estimated_memory_gb: float = 0


class ExecutorResult(BaseModel):
    """Result from the Executor Agent."""

    iteration: int
    success: bool
    training_result: Optional[TrainingResult] = None
    error_message: Optional[str] = None

    # Analysis
    analysis: str = Field(default="", description="Analysis of the training run")
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for next iteration"
    )

    # Performance relative to previous
    improvement: Optional[float] = None
    is_best_so_far: bool = False


class ExperimentRecord(BaseModel):
    """Record of a single experiment iteration."""

    id: Optional[int] = None
    experiment_id: str
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    dataset_info: DatasetInfo
    recommendation: AnalyzerRecommendation
    result: ExecutorResult

    # Budget tracking
    api_calls_used: int = 0
    compute_time_seconds: float = 0


class OrchestratorState(BaseModel):
    """Complete state of the orchestrator for checkpointing."""

    experiment_id: str
    dataset_info: DatasetInfo

    # Budget
    total_budget: int
    remaining_budget: int

    # Progress
    current_iteration: int
    total_iterations: int

    # History
    experiment_history: list[ExperimentRecord] = Field(default_factory=list)

    # Best results
    best_result: Optional[ExecutorResult] = None
    best_config: Optional[TrainingConfig] = None
    best_metric_value: float = 0

    # Timing
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_checkpoint_time: Optional[datetime] = None

    # Status
    status: str = Field(
        default="initialized"
    )  # initialized, running, paused, completed, failed
    error_message: Optional[str] = None
