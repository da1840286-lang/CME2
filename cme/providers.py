"""Paid models only, explicit ids, no silent fallback, every call logged."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path

AMBIGUOUS_MODELS = ("openrouter/free",)  # routers that pick an arbitrary model
TIERS = ("paid", "free")


class ProviderPolicyError(Exception):
    pass


class ModelMismatch(ProviderPolicyError):
    """The model that answered is not the one that was asked for."""


class InfraError(Exception):
    """Transient provider failure (rate limit, quota, timeout). Not a quality failure."""


class ProviderCallError(Exception):
    """The provider answered, but the call failed for a non-transient reason."""


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model_id: str
    tier: str = "paid"  # "paid" or "free"; free must be declared, never inferred


def check_model(ref: ModelRef, allow_free: bool = False) -> ModelRef:
    """Free adapters are allowed only when declared as free AND the run explicitly allows free."""
    low = (ref.model_id or "").lower()
    if not low:
        raise ProviderPolicyError("model id is required")
    if any(m in low for m in AMBIGUOUS_MODELS):
        raise ProviderPolicyError(f"router that picks an arbitrary model is not allowed: {ref.model_id!r}")
    if ref.tier not in TIERS:
        raise ProviderPolicyError(f"unknown tier {ref.tier!r}")
    if ":free" in low and ref.tier != "free":
        raise ProviderPolicyError(f"{ref.model_id!r} looks free but is declared {ref.tier!r}")
    if ref.tier == "free" and not allow_free:
        raise ProviderPolicyError(f"free adapter {ref.model_id!r} needs allow_free=True in the run settings")
    return ref


def require_paid(ref: ModelRef) -> ModelRef:
    return check_model(ref, allow_free=False)


def _base_model(model_id: str) -> str:
    """Compare model ids without the routing variant suffix (the part after a colon)."""
    return str(model_id).split(":", 1)[0].strip().lower()


class CallLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record) -> None:
        record.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_with_policy(primary: ModelRef, fn, log: CallLog, run_id: str, fallback: ModelRef | None = None,
                     allow_free: bool = False):
    """fn(model_ref) -> (result, meta). Fallback only on InfraError and only if declared."""
    check_model(primary, allow_free)
    if fallback:
        check_model(fallback, allow_free)
    for ref, is_fallback in ((primary, False), (fallback, True)):
        if ref is None:
            break
        try:
            result, meta = fn(ref)
            responded = meta.get("model")
            if responded and _base_model(responded) != _base_model(ref.model_id):
                log.write(run_id=run_id, requested_model=ref.model_id, responded_model=responded,
                          provider=ref.provider, tier=ref.tier, status="model_mismatch", fallback_used=is_fallback)
                raise ModelMismatch(f"asked for {ref.model_id!r} but {responded!r} answered")
            record = {
                "run_id": run_id,
                "requested_model": primary.model_id,
                "responded_model": responded or "unreported",
                "provider": ref.provider,
                "tier": ref.tier,
                "cost": meta.get("cost"),
                "status": "ok",
                "fallback_used": is_fallback,
            }
            # Provider adapters may attach bounded, redacted diagnostics for a structurally invalid response.
            if meta.get("diagnostic") is not None:
                record["diagnostic"] = meta["diagnostic"]
            log.write(**record)
            return result
        except InfraError as e:
            log.write(run_id=run_id, requested_model=ref.model_id, status="infra_error", error=str(e),
                      fallback_used=is_fallback)
            if is_fallback or fallback is None:
                raise
        except ModelMismatch:
            raise  # already logged above
        except Exception as e:  # real failures are logged too, then raised
            log.write(run_id=run_id, requested_model=ref.model_id, provider=ref.provider, tier=ref.tier,
                      status="error", error_type=type(e).__name__, error=str(e), fallback_used=is_fallback)
            raise
    raise InfraError("no provider available")
