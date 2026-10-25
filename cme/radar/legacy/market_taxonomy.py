"""Market taxonomies loaded from ``config/verticals/*.json``.

A vertical is configuration, not code: drop a JSON file into ``config/verticals`` (or point
``CME_VERTICALS_DIR`` elsewhere) and the discovery, source-verification and transfer layers pick it
up. See ``config/verticals/physical_products.json`` for a minimal e-commerce example.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarketTaxonomy:
    name: str
    direct_brands: tuple[str, ...]
    direct_domains: tuple[str, ...]
    direct_url_fragments: tuple[str, ...]
    direct_terms: tuple[str, ...]
    adjacent_brands: tuple[str, ...]
    adjacent_terms: tuple[str, ...]
    hard_exclude_brands: tuple[str, ...]
    hard_exclude_domains: tuple[str, ...]
    hard_exclude_terms: tuple[str, ...]
    seed_groups: tuple[tuple[str, ...], ...]
    context_required_brands: tuple[str, ...] = ()
    # Search-only ambiguity rules. These do not alter source-market classification.
    ambiguous_seed_terms: tuple[str, ...] = ()
    # Entity-level context rules are generic taxonomy data, not brand-specific code.
    # Each tuple is (entity, positive_context_terms, negative_context_terms).
    entity_context_rules: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
    notes: str = ""
    aliases: tuple[str, ...] = ()
    # Only used to decide whether the ambiguous operator label "Software" means this vertical.
    software_alias_intent_terms: tuple[str, ...] = ()
    # Native-industry labels that conflict with this vertical (weak prior, needs corroboration).
    conflict_industries: tuple[str, ...] = ()
    profile_include_extra: tuple[str, ...] = ()
    profile_topical: tuple[str, ...] = ()
    target_context: dict[str, str] = field(default_factory=dict)


def norm_key(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _tup(value: Any) -> tuple:
    return tuple(value or ())


def _from_json(data: dict[str, Any]) -> MarketTaxonomy:
    return MarketTaxonomy(
        name=str(data["name"]),
        direct_brands=_tup(data.get("direct_brands")),
        direct_domains=_tup(data.get("direct_domains")),
        direct_url_fragments=_tup(data.get("direct_url_fragments")),
        direct_terms=_tup(data.get("direct_terms")),
        adjacent_brands=_tup(data.get("adjacent_brands")),
        adjacent_terms=_tup(data.get("adjacent_terms")),
        hard_exclude_brands=_tup(data.get("hard_exclude_brands")),
        hard_exclude_domains=_tup(data.get("hard_exclude_domains")),
        hard_exclude_terms=_tup(data.get("hard_exclude_terms")),
        seed_groups=tuple(tuple(g) for g in (data.get("seed_groups") or ())),
        context_required_brands=_tup(data.get("context_required_brands")),
        ambiguous_seed_terms=_tup(data.get("ambiguous_seed_terms")),
        entity_context_rules=tuple(
            (str(e), tuple(pos), tuple(neg)) for e, pos, neg in (data.get("entity_context_rules") or ())
        ),
        notes=str(data.get("notes") or ""),
        aliases=_tup(data.get("aliases")),
        software_alias_intent_terms=_tup(data.get("software_alias_intent_terms")),
        conflict_industries=_tup(data.get("conflict_industries")),
        profile_include_extra=_tup(data.get("profile_include_extra")),
        profile_topical=_tup(data.get("profile_topical")),
        target_context=dict(data.get("target_context") or {}),
    )


def verticals_dir() -> Path:
    env = os.getenv("CME_VERTICALS_DIR", "").strip()
    return Path(env) if env else Path(__file__).resolve().parents[1] / "config" / "verticals"


def load_taxonomies(directory: str | Path | None = None) -> dict[str, MarketTaxonomy]:
    base = Path(directory) if directory else verticals_dir()
    if not base.is_dir():
        raise FileNotFoundError(f"Vertical config directory not found: {base}")
    found: dict[str, MarketTaxonomy] = {}
    for path in sorted(base.glob("*.json")):
        try:
            tax = _from_json(json.loads(path.read_text(encoding="utf-8")))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid vertical config {path.name}: {exc}") from exc
        key = norm_key(tax.name)
        if key in found:
            raise ValueError(f"Duplicate vertical name {tax.name!r} in {path.name}")
        found[key] = tax
    if not found:
        raise ValueError(f"No vertical configs (*.json) in {base}")
    return found


TAXONOMIES: dict[str, MarketTaxonomy] = load_taxonomies()


def _build_aliases(taxonomies: dict[str, MarketTaxonomy]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, tax in taxonomies.items():
        out[key] = key
        for alias in tax.aliases:
            a = norm_key(alias)
            if a in out and out[a] != key:
                raise ValueError(f"Vertical alias {alias!r} is claimed by two verticals")
            out[a] = key
    return out


ALIASES: dict[str, str] = _build_aliases(TAXONOMIES)


def resolve_key(vertical: str) -> str | None:
    """Return the canonical taxonomy key for a vertical name or alias, else None."""
    return ALIASES.get(norm_key(vertical))


def default_vertical_name() -> str:
    """Operator default; set CME_DEFAULT_VERTICAL to switch the whole system to another vertical."""
    configured = os.getenv("CME_DEFAULT_VERTICAL", "").strip()
    if configured:
        key = resolve_key(configured)
        if key is None:
            raise ValueError(f"CME_DEFAULT_VERTICAL={configured!r} is not a configured vertical: {sorted(TAXONOMIES)}")
        return TAXONOMIES[key].name
    return TAXONOMIES["ai developer tools"].name


def term_in_text(text: str, term: str) -> bool:
    """Whole-word / whole-phrase match (no substring hits like 'ai' inside 'hair')."""
    text = " ".join((text or "").lower().split())
    term = " ".join((term or "").lower().split())
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


AI_DEVELOPER_TOOLS = TAXONOMIES["ai developer tools"]  # backward-compatible alias
