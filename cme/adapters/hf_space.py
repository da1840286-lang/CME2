"""Hugging Face Space image adapter (free tier, ZeroGPU). Space specifics live in config, not here."""
from __future__ import annotations
import inspect
from pathlib import Path
from ..providers import InfraError, ProviderCallError

QUOTA_MARKERS = ("quota", "zerogpu", "gpu abort", "exceeded your", "rate limit", "too many requests")


def default_client_factory(space_id: str, token: str | None):
    """Works across gradio_client releases that take `token` or `hf_token`."""
    from gradio_client import Client, handle_file
    params = inspect.signature(Client).parameters
    kwargs = {"verbose": False}
    if token:
        if "token" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs["token"] = token
        elif "hf_token" in params:
            kwargs["hf_token"] = token
        else:
            raise ProviderCallError("gradio_client accepts neither token nor hf_token")
    return Client(space_id, **kwargs), handle_file


def extract_image_path(result) -> Path:
    """Gradio returns a path, a dict with 'path', or nested lists/tuples of those."""
    if isinstance(result, (str, Path)):
        p = Path(result)
        if p.exists():
            return p
    elif isinstance(result, dict):
        for key in ("path", "image", "value"):
            if key in result:
                try:
                    return extract_image_path(result[key])
                except ProviderCallError:
                    pass
    elif isinstance(result, (list, tuple)):
        for item in result:
            try:
                return extract_image_path(item)
            except ProviderCallError:
                continue
    raise ProviderCallError(f"no image file found in Space result: {str(result)[:120]}")


def make_hf_space_image(params: dict, token: str | None = None, client_factory=None):
    """params: space_id, api_name, args (ordered mapping), width, height, seed."""
    factory = client_factory or default_client_factory

    def generate(prompt: str, refs) -> bytes:
        try:
            client, handle_file = factory(params["space_id"], token)
            args = []
            for a in params["args"]:
                if "const" in a:
                    args.append(a["const"])
                    continue
                src = a["from"]
                if src == "prompt":
                    args.append(prompt)
                elif src in ("width", "height", "seed"):
                    args.append(params.get(src, 0))
                elif src == "refs_gallery":
                    # Gradio Gallery input: [{image: FileData, caption: None}], order matters
                    # (preceding shot first, then the other references)
                    gal = [{"image": handle_file(str(r)), "caption": None} for r in refs]
                    args.append(gal if gal else a.get("default"))
                elif src == "ref":
                    i = a.get("index", 0)
                    args.append(handle_file(str(refs[i])) if i < len(refs) else a.get("default"))
                else:
                    raise ProviderCallError(f"unknown arg source {src!r}")
            result = client.predict(*args, api_name=params["api_name"])
            return extract_image_path(result).read_bytes()
        except ProviderCallError:
            raise
        except Exception as e:  # classify: quota is infrastructure, everything else is a real failure
            msg = str(e).lower()
            if any(m in msg for m in QUOTA_MARKERS):
                raise InfraError(f"Space quota or availability: {e}") from e
            raise ProviderCallError(str(e)) from e

    return generate
