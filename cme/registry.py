"""Adapter catalog + run settings: choose which adapter to test, explicitly.

Free and paid adapters sit side by side. Nothing falls back silently, and an adapter whose
parameters were never confirmed by a live call runs only when the settings opt in.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from .adapters.fal_http import make_fal_image
from .adapters.hf_space import make_hf_space_image
from .adapters.openrouter import make_openrouter_analyst, make_openrouter_brain, make_openrouter_judge
from .providers import CallLog, InfraError, ModelRef, ProviderPolicyError, call_with_policy, check_model

class SettingsError(ValueError):
    pass


FACTORIES = {
    ("hf_space", "image"): lambda params, **ctx: make_hf_space_image(
        {**params, **({"seed": ctx["image_seed"]} if ctx.get("image_seed") is not None else {})},
        token=ctx.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        client_factory=ctx.get("client_factory")),
    ("fal", "image"): lambda params, **ctx: make_fal_image(params, api_key=ctx.get("fal_key"), transport=ctx.get("transport")),
    ("openrouter", "brain"): lambda params, **ctx: make_openrouter_brain(
        params, api_key=ctx.get("openrouter_key"), transport=ctx.get("transport")),
    ("openrouter", "judge"): lambda params, **ctx: make_openrouter_judge(
        params, api_key=ctx.get("openrouter_key"), transport=ctx.get("transport")),
    ("openrouter", "analyst"): lambda params, **ctx: make_openrouter_analyst(
        params, api_key=ctx.get("openrouter_key"), transport=ctx.get("transport")),
}


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    kind: str          # image / brain / judge / analyst
    provider: str
    model_id: str
    tier: str          # paid / free
    params: dict = field(default_factory=dict)
    verified: bool = False
    price_hint: str = ""
    notes: str = ""

    @property
    def ref(self) -> ModelRef:
        return ModelRef(self.provider, self.model_id, self.tier)


@dataclass(frozen=True)
class Settings:
    image_adapter: str = ""
    fallback_adapter: str = ""        # image fallback, must be declared to be used
    allow_free: bool = False
    brain_adapter: str = ""
    judge_adapter: str = ""
    allow_unverified: bool = False    # first live test of a freshly configured adapter
    max_attempts_per_shot: int = 3
    analyst_adapter: str = ""          # Creative Radar: manual ad analysis
    # Resilience for Brain / Target-Fit Gate / Judge. Only transient infrastructure errors (429, 5xx, network)
    # are retried; malformed or negative model answers are never retried here.
    brain_fallback_adapter: str = ""   # optional reserve model of kind "brain", used after retries are exhausted
    judge_fallback_adapter: str = ""   # optional reserve model of kind "judge"
    transient_retries: int = 1         # extra attempts on the same model after a transient error (0-3)
    retry_delay_seconds: float = 0.0   # pause before a retry; 0 keeps offline tests instant
    # Source selection. Hard conditions (video, scorer not REJECT, no hard-exclude) always apply. The values
    # below are optional and OFF by default: a weak source is flagged WEAK_SOURCE, or blocked in "enforce" mode.
    source_min_final_score: float | None = None
    source_allowed_actions: tuple = ()
    source_threshold_mode: str = "flag"  # "flag" (mark WEAK_SOURCE) | "enforce" (NO_QUALIFIED_SOURCE)


def load_catalog(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["name"]: AdapterSpec(e["name"], e["kind"], e["provider"], e["model_id"], e["tier"],
                                   e.get("params", {}), bool(e.get("verified", False)),
                                   e.get("price_hint", ""), e.get("notes", "")) for e in data["adapters"]}


def load_settings(path) -> Settings:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SettingsError(f"cannot read settings JSON: {e}") from e
    if not isinstance(d, dict):
        raise SettingsError("settings.json root must be a JSON object")
    for key in ("allow_free", "allow_unverified"):
        value = d.get(key, False)
        if type(value) is not bool:
            raise SettingsError(f"{key} must be a JSON boolean (true/false), not {value!r}")
    attempts = d.get("max_attempts_per_shot", 3)
    if type(attempts) is not int or attempts < 1:
        raise SettingsError("max_attempts_per_shot must be a positive integer")
    names = ("image_adapter", "fallback_adapter", "brain_adapter", "judge_adapter", "analyst_adapter",
             "brain_fallback_adapter", "judge_fallback_adapter")
    for key in names:
        if key in d and not isinstance(d[key], str):
            raise SettingsError(f"{key} must be a string")
    retries = d.get("transient_retries", 1)
    if type(retries) is not int or not 0 <= retries <= 3:
        raise SettingsError("transient_retries must be an integer from 0 to 3")
    delay = d.get("retry_delay_seconds", 0.0)
    if type(delay) not in (int, float) or not 0 <= delay <= 30:
        raise SettingsError("retry_delay_seconds must be a number from 0 to 30")
    selection = d.get("source_selection", {})
    if not isinstance(selection, dict):
        raise SettingsError("source_selection must be a JSON object")
    min_score = selection.get("min_final_score")
    if min_score is not None and (type(min_score) not in (int, float)):
        raise SettingsError("source_selection.min_final_score must be a number or null")
    actions = selection.get("allowed_actions", [])
    if not isinstance(actions, list) or any(not isinstance(a, str) or not a.strip() for a in actions):
        raise SettingsError("source_selection.allowed_actions must be a list of non-empty strings")
    mode = selection.get("mode", "flag")
    if mode not in ("flag", "enforce"):
        raise SettingsError("source_selection.mode must be 'flag' or 'enforce'")
    return Settings(image_adapter=d.get("image_adapter", ""), fallback_adapter=d.get("fallback_adapter", ""),
                    allow_free=d.get("allow_free", False), brain_adapter=d.get("brain_adapter", ""),
                    judge_adapter=d.get("judge_adapter", ""), allow_unverified=d.get("allow_unverified", False),
                    max_attempts_per_shot=attempts, analyst_adapter=d.get("analyst_adapter", ""),
                    brain_fallback_adapter=d.get("brain_fallback_adapter", ""),
                    judge_fallback_adapter=d.get("judge_fallback_adapter", ""),
                    transient_retries=retries, retry_delay_seconds=float(delay),
                    source_min_final_score=None if min_score is None else float(min_score),
                    source_allowed_actions=tuple(a.strip().upper() for a in actions),
                    source_threshold_mode=mode)


def describe(catalog: dict) -> str:
    rows = ["name | kind | tier | verified | price hint"]
    for s in catalog.values():
        rows.append(f"{s.name} | {s.kind} | {s.tier} | {'yes' if s.verified else 'NO (confirm on first live run)'} | {s.price_hint}")
    return "\n".join(rows)


def build_adapter(spec: AdapterSpec, allow_free: bool = False, allow_unverified: bool = False, **ctx):
    check_model(spec.ref, allow_free)
    if "CONFIRM_ME" in json.dumps(spec.params):
        raise ProviderPolicyError(f"adapter {spec.name!r} is not configured: replace CONFIRM_ME in adapters.json")
    if not spec.verified and not allow_unverified:
        raise ProviderPolicyError(
            f"adapter {spec.name!r} was never confirmed by a live call: set allow_unverified for the first test only")
    factory = FACTORIES.get((spec.provider, spec.kind))
    if factory is None:
        raise ProviderPolicyError(f"no factory for {spec.provider!r}/{spec.kind!r}")
    return factory({**spec.params, "model_id": spec.model_id}, **ctx)


def _info(spec, primary, image_seed=None):
    info = {"adapter": spec.name, "tier": spec.tier, "model_id": spec.model_id, "provider": spec.provider,
            "verified": spec.verified, "fallback_used": spec.name != primary.name}
    if spec.kind == "image" and spec.provider == "hf_space" and image_seed is not None:
        info["seed"] = image_seed
    return info


def make_image_provider(settings: Settings, catalog: dict, log: CallLog, run_id: str, **ctx):
    """provider(prompt, refs) -> bytes. `provider.info` describes the last call for the receipt."""
    if settings.image_adapter not in catalog:
        raise ProviderPolicyError(f"unknown image_adapter {settings.image_adapter!r} in settings.json")
    if settings.fallback_adapter and settings.fallback_adapter not in catalog:
        raise ProviderPolicyError(f"unknown fallback_adapter {settings.fallback_adapter!r} in settings.json")
    primary = catalog[settings.image_adapter]
    fb = catalog[settings.fallback_adapter] if settings.fallback_adapter else None
    image_seed = ctx.get("image_seed")
    specs = {primary.ref: primary, **({fb.ref: fb} if fb else {})}
    fns = {s.name: build_adapter(s, settings.allow_free, settings.allow_unverified, **ctx) for s in specs.values()}

    def provider(prompt: str, refs) -> bytes:
        def go(ref):
            spec = specs[ref]
            provider.info = _info(spec, primary, image_seed) | {"call_status": "STARTED"}
            try:
                data = fns[spec.name](prompt, refs)
            except Exception as e:
                provider.info = _info(spec, primary, image_seed) | {"call_status": "ERROR",
                    "error_type": type(e).__name__, "error": str(e)[:300]}
                raise
            provider.info = _info(spec, primary, image_seed) | {"call_status": "OK"}
            return data, {}  # image services do not report which model ran: logged as 'unreported'
        return call_with_policy(primary.ref, go, log, run_id, fb.ref if fb else None, settings.allow_free)

    provider.info = {}
    return provider


# Kinds whose transient infrastructure errors are retried on the same model. The Target-Fit Gate shares the
# Brain call, so it is covered by "brain". Analyst failures are handled by the Radar source loop instead.
_RETRY_KINDS = {"brain", "judge"}
_sleep = time.sleep  # module-level so tests can replace it


def _fallback_spec(kind, primary, fallback_name, catalog):
    if not fallback_name:
        return None
    if fallback_name not in catalog:
        raise ProviderPolicyError(f"unknown {kind}_fallback_adapter {fallback_name!r} in settings.json")
    spec = catalog[fallback_name]
    if spec.kind != kind:
        raise ProviderPolicyError(f"{kind}_fallback_adapter {fallback_name!r} is a {spec.kind!r} adapter, not {kind!r}")
    if spec.name == primary.name or spec.model_id == primary.model_id:
        raise ProviderPolicyError(f"{kind}_fallback_adapter must be a different model from the primary adapter")
    return spec


def _make_llm(adapter_name, kind, settings, catalog, log, run_id, fallback_name="", **ctx):
    if not adapter_name:
        raise ProviderPolicyError(f"{kind}_adapter is not configured in settings.json")
    if adapter_name not in catalog:
        raise ProviderPolicyError(f"unknown {kind}_adapter {adapter_name!r} in settings.json")
    spec = catalog[adapter_name]
    if spec.kind != kind:
        raise ProviderPolicyError(f"adapter {adapter_name!r} is a {spec.kind!r} adapter, not {kind!r}")
    fn = build_adapter(spec, settings.allow_free, settings.allow_unverified, **ctx)
    fb_spec = _fallback_spec(kind, spec, fallback_name, catalog)
    fb_fn = build_adapter(fb_spec, settings.allow_free, settings.allow_unverified, **ctx) if fb_spec else None
    retrying = kind in _RETRY_KINDS
    attempts = 1 + max(0, int(getattr(settings, "transient_retries", 1))) if retrying else 1
    delay = float(getattr(settings, "retry_delay_seconds", 0.0) or 0.0)
    component = "qa_judge_resilience" if kind == "judge" else f"{kind}_resilience"

    def run(*args):
        def make_go(used_spec, used_fn):
            def go(ref):
                out, meta = used_fn(*args)
                run.info = _info(used_spec, spec) | {"responded_model": meta.get("model")}
                return out, meta
            return go

        # Retry only a transient infrastructure failure on the same explicit model.
        # Never retry malformed/negative answers, policy errors, or non-transient provider errors.
        last = None
        for attempt in range(1, attempts + 1):
            try:
                return call_with_policy(spec.ref, make_go(spec, fn), log, run_id, allow_free=settings.allow_free)
            except InfraError as e:
                last = e
                if not retrying:
                    raise
                final = attempt == attempts
                log.write(run_id=run_id, component=component, attempt=attempt, max_attempts=attempts,
                          status="exhausted" if final else "retry", error=str(e)[:300])
                if not final and delay > 0:
                    _sleep(delay)
        if fb_spec is not None:
            # The reserve model is declared in settings; it is used only after the primary's retries are exhausted.
            log.write(run_id=run_id, component=f"{kind}_fallback", status="switching",
                      from_model=spec.model_id, to_model=fb_spec.model_id, error=str(last)[:300])
            return call_with_policy(fb_spec.ref, make_go(fb_spec, fb_fn), log, run_id, allow_free=settings.allow_free)
        raise last

    run.info = {}
    return run


def make_brain_call(settings, catalog, log, run_id, **ctx):
    """call(request) -> dict, for brain.request_atoms."""
    return _make_llm(settings.brain_adapter, "brain", settings, catalog, log, run_id,
                     fallback_name=settings.brain_fallback_adapter, **ctx)


def make_judge(settings, catalog, log, run_id, **ctx):
    """judge(image_path, atoms) -> {atom_id: {pass, evidence}}, for qa.judge_batch_checks."""
    return _make_llm(settings.judge_adapter, "judge", settings, catalog, log, run_id,
                     fallback_name=settings.judge_fallback_adapter, **ctx)


def make_analyst(settings, catalog, log, run_id, **ctx):
    """analyst(frames, context) -> ({'data', 'raw_excerpt'}), for the Creative Radar manual analysis."""
    return _make_llm(settings.analyst_adapter, "analyst", settings, catalog, log, run_id, **ctx)
