import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.models import (
    DatasetInfo,
    ExperimentRecord,
    OrchestratorState,
    AnalyzerRecommendation,
    ExecutorResult,
)
from src.utils import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class ExperimentTable(Base):
    """SQLAlchemy model for experiments."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(100), index=True)
    iteration = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Serialized data
    dataset_info_json = Column(Text)
    recommendation_json = Column(Text)
    result_json = Column(Text)

    # Metrics for quick querying
    primary_metric_value = Column(Float)
    backbone = Column(String(100))
    strategy = Column(String(50))

    # Budget tracking
    api_calls_used = Column(Integer, default=0)
    compute_time_seconds = Column(Float, default=0)


class ExperimentTracker:
    """
    Tracks experiment history with database persistence.
    """

    def __init__(self, database_url: str = "sqlite:///experiments/experiments.db"):
        """Initialize experiment tracker."""
        # Ensure directory exists
        db_path = database_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine)

        logger.info(f"ExperimentTracker initialized with database: {database_url}")

    def record_experiment(self, record: ExperimentRecord) -> int:
        """
        Record an experiment iteration.

        Returns:
            Database ID of the record
        """
        session = self.Session()

        try:
            db_record = ExperimentTable(
                experiment_id=record.experiment_id,
                iteration=record.iteration,
                timestamp=record.timestamp,
                dataset_info_json=record.dataset_info.model_dump_json(),
                recommendation_json=record.recommendation.model_dump_json(),
                result_json=record.result.model_dump_json(),
                primary_metric_value=(
                    record.result.training_result.primary_metric_value
                    if record.result.training_result
                    else None
                ),
                backbone=(record.recommendation.training_config.backbone.full_name),
                strategy=(
                    record.recommendation.training_config.strategy.strategy_type.value
                ),
                api_calls_used=record.api_calls_used,
                compute_time_seconds=record.compute_time_seconds,
            )

            session.add(db_record)
            session.commit()

            record_id = db_record.id
            logger.debug(
                f"Recorded experiment iteration {record.iteration}, ID: {record_id}"
            )

            return record_id

        finally:
            session.close()

    def get_experiment_history(
        self,
        experiment_id: str,
        limit: Optional[int] = None,
    ) -> List[ExperimentRecord]:
        """Get experiment history for an experiment ID."""
        session = self.Session()

        try:
            query = (
                session.query(ExperimentTable)
                .filter(ExperimentTable.experiment_id == experiment_id)
                .order_by(ExperimentTable.iteration.asc())
            )

            if limit:
                query = query.limit(limit)

            records = []
            for db_record in query.all():
                record = ExperimentRecord(
                    id=db_record.id,
                    experiment_id=db_record.experiment_id,
                    iteration=db_record.iteration,
                    timestamp=db_record.timestamp,
                    dataset_info=DatasetInfo.model_validate_json(
                        db_record.dataset_info_json
                    ),
                    recommendation=AnalyzerRecommendation.model_validate_json(
                        db_record.recommendation_json
                    ),
                    result=ExecutorResult.model_validate_json(db_record.result_json),
                    api_calls_used=db_record.api_calls_used,
                    compute_time_seconds=db_record.compute_time_seconds,
                )
                records.append(record)

            return records

        finally:
            session.close()

    def get_best_result(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Get the best result for an experiment."""
        session = self.Session()

        try:
            db_record = (
                session.query(ExperimentTable)
                .filter(
                    ExperimentTable.experiment_id == experiment_id,
                    ExperimentTable.primary_metric_value.isnot(None),
                )
                .order_by(ExperimentTable.primary_metric_value.desc())
                .first()
            )

            if db_record is None:
                return None

            return ExperimentRecord(
                id=db_record.id,
                experiment_id=db_record.experiment_id,
                iteration=db_record.iteration,
                timestamp=db_record.timestamp,
                dataset_info=DatasetInfo.model_validate_json(
                    db_record.dataset_info_json
                ),
                recommendation=AnalyzerRecommendation.model_validate_json(
                    db_record.recommendation_json
                ),
                result=ExecutorResult.model_validate_json(db_record.result_json),
                api_calls_used=db_record.api_calls_used,
                compute_time_seconds=db_record.compute_time_seconds,
            )

        finally:
            session.close()

    def get_all_experiments(self) -> List[str]:
        """Get all unique experiment IDs."""
        session = self.Session()

        try:
            results = session.query(ExperimentTable.experiment_id).distinct().all()
            return [r[0] for r in results]
        finally:
            session.close()

    def get_experiment_summary(self, experiment_id: str) -> dict:
        """Get summary statistics for an experiment."""
        session = self.Session()

        try:
            records = (
                session.query(ExperimentTable)
                .filter(ExperimentTable.experiment_id == experiment_id)
                .all()
            )

            if not records:
                return {}

            metrics = [
                r.primary_metric_value for r in records if r.primary_metric_value
            ]

            return {
                "experiment_id": experiment_id,
                "total_iterations": len(records),
                "best_metric": max(metrics) if metrics else None,
                "worst_metric": min(metrics) if metrics else None,
                "avg_metric": sum(metrics) / len(metrics) if metrics else None,
                "total_api_calls": sum(r.api_calls_used for r in records),
                "total_compute_time": sum(r.compute_time_seconds for r in records),
                "unique_backbones": list(set(r.backbone for r in records)),
                "unique_strategies": list(set(r.strategy for r in records)),
            }

        finally:
            session.close()
