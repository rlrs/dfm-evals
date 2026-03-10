# Modal Build Context Patch

## Problem

Some Harbor tasks fail during Modal image builds with errors like:

- `copy_tree: source path does not exist`
- `COPY ./data /workdir/data`
- `COPY api_bug.py /app/api_bug.py`

The Harbor task itself is often fine. The breakage comes from the Modal adapter losing the Docker build context.

## Root Cause

Harbor passes each task environment to Inspect as a `ComposeBuild(context=...)` object, pointing at the task's `environment/` directory.

`inspect_sandboxes.modal._compose.convert_compose_to_modal_params()` was calling:

```python
modal.Image.from_dockerfile(str(dockerfile_path))
```

without `context_dir=...`.

When `context_dir` is omitted, the Modal SDK falls back to the current working directory. That breaks any Harbor Dockerfile that uses relative `COPY` sources from its own task environment.

## Local Fix

`dfm_evals/cli.py` now patches `inspect_sandboxes` at runtime so Modal image builds use:

```python
modal.Image.from_dockerfile(
    str(dockerfile_path),
    context_dir=str(build_context_dir),
)
```

The patch is applied automatically when the CLI loads task registries.

## Upstreaming

This should be fixed upstream in `meridianlabs-ai/inspect_sandboxes`, not kept
as a long-term local monkey patch.

The relevant upstream path is the Modal compose adapter:

- `inspect_sandboxes.modal._compose.convert_compose_to_modal_params()`

That code already resolves the Dockerfile path, but it also needs to pass the
matching build context to:

```python
modal.Image.from_dockerfile(..., context_dir=...)
```

The context should come from Harbor's `ComposeBuild.context` (or `"."` when it
is unset), resolved relative to the compose file directory.

Once `inspect_sandboxes` releases that fix, the local patch in
`dfm_evals/cli.py` can be removed.

## Why This Matters

This is not just a single-task issue. Many Harbor tasks rely on local build context files such as:

- `COPY data /workdir/data`
- `COPY src/ /app/service/src/`
- `COPY api_bug.py /app/api_bug.py`

Without the correct build context, those tasks fail before the model even starts solving them.
