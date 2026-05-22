"""AmyDPOTrainer - TRL DPOTrainer subclass for AmyLM DPO with FACodec."""

from __future__ import annotations

from trl import DPOConfig, DPOTrainer


class AmyDPOTrainer(DPOTrainer):
    """DPOTrainer subclass for AmyLM DPO training.

    Key differences from standard DPOTrainer:
     - precompute_ref_log_probs=True by default (lambda-zero at init)
     - Logs lambda_p and lambda_t from AmyLM's ResidualFusion gates
     - Passes AmyLM-specific kwargs (audio_data, prosody_indices, etc.)
       through to model.forward() automatically via TRL's model_kwargs passthrough
    """

    def __init__(
        self,
        model,
        ref_model=None,
        args: DPOConfig | None = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        **kwargs,
    ):
        if args is None:
            args = DPOConfig(
                output_dir="./amy_dpo_output",
                precompute_ref_log_probs=True,
                loss_type=["sigmoid"],
            )
        elif not args.precompute_ref_log_probs:
            raise ValueError(
                "AmyDPOTrainer requires precompute_ref_log_probs=True. "
                "This ensures the reference model (lambda=0) is computed "
                "before training starts, when FACodec gates are zero-initialized."
            )

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs,
        )

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        """Inject lambda_p and lambda_t into TRL's log output."""
        try:
            model = self.accelerator.unwrap_model(self.model)
            base_model = getattr(model, "base_model", model)
            fusion = base_model.residual_fusion
            logs["lambda_p"] = float(fusion.lambda_p.item())
            logs["lambda_t"] = float(fusion.lambda_t.item())
        except AttributeError:
            pass
        super().log(logs, *args, **kwargs)
