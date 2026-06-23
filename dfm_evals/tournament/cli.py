import argparse
from typing import Any, Callable, Sequence, TypeVar

ResultT = TypeVar("ResultT")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "init":
        from .store import initialize_tournament_store

        db_path = initialize_tournament_store(args.config)
        print(db_path)
        return 0

    if args.command == "generate":
        from ._cli_format import format_generation_result, generation_result_payload
        from .generation import run_generation

        models = args.models if args.models else None
        return _emit_result(
            run_generation(args.config, models=models),
            payload_fn=generation_result_payload,
            formatter_fn=format_generation_result,
            json_out=args.json_out,
        )

    if args.command == "run":
        from ._cli_format import format_run_result, run_result_payload
        from .orchestrator import run_tournament

        return _emit_result(
            run_tournament(args.config, max_batches=args.max_batches),
            payload_fn=run_result_payload,
            formatter_fn=format_run_result,
            json_out=args.json_out,
        )

    if args.command == "resume":
        from ._cli_format import format_run_result, run_result_payload
        from .orchestrator import resume_tournament

        return _emit_result(
            resume_tournament(args.target, max_batches=args.max_batches),
            payload_fn=run_result_payload,
            formatter_fn=format_run_result,
            json_out=args.json_out,
        )

    if args.command == "add-model":
        from ._cli_format import add_models_result_payload, format_add_models_result
        from .orchestrator import add_models

        return _emit_result(
            add_models(
                args.target,
                models=args.models,
                max_batches=args.max_batches,
            ),
            payload_fn=add_models_result_payload,
            formatter_fn=format_add_models_result,
            json_out=args.json_out,
        )

    if args.command == "register-model":
        from ._cli_format import format_register_models_result, register_models_result_payload
        from .orchestrator import register_models

        return _emit_result(
            register_models(
                args.target,
                models=args.models,
            ),
            payload_fn=register_models_result_payload,
            formatter_fn=format_register_models_result,
            json_out=args.json_out,
        )

    if args.command == "update-config":
        from ._cli_format import format_update_config_result, update_config_result_payload
        from .orchestrator import update_tournament_config

        return _emit_result(
            update_tournament_config(
                args.target,
                max_total_matches=args.max_total_matches,
            ),
            payload_fn=update_config_result_payload,
            formatter_fn=format_update_config_result,
            json_out=args.json_out,
        )

    if args.command == "status":
        from ._cli_format import format_status, status_payload
        from .orchestrator import tournament_status

        return _emit_result(
            tournament_status(args.target),
            payload_fn=status_payload,
            formatter_fn=format_status,
            json_out=args.json_out,
        )

    if args.command == "export":
        from ._cli_format import export_result_payload, format_export_result
        from .exports import export_rankings

        return _emit_result(
            export_rankings(args.target, output_dir=args.output_dir),
            payload_fn=export_result_payload,
            formatter_fn=format_export_result,
            json_out=args.json_out,
        )

    if args.command == "export-prompts":
        from ._cli_format import (
            format_prompt_responses_export_result,
            prompt_responses_export_result_payload,
        )
        from .exports import export_prompt_responses

        return _emit_result(
            export_prompt_responses(args.target, output_path=args.output),
            payload_fn=prompt_responses_export_result_payload,
            formatter_fn=format_prompt_responses_export_result,
            json_out=args.json_out,
        )

    if args.command == "export-html":
        from ._cli_format import format_view_export_result, view_export_result_payload
        from .viewer import export_tournament_view_html

        return _emit_result(
            export_tournament_view_html(args.target, output_path=args.output),
            payload_fn=view_export_result_payload,
            formatter_fn=format_view_export_result,
            json_out=args.json_out,
        )

    if args.command == "view":
        from .viewer import (
            format_tournament_view_runs,
            list_tournament_view_runs,
            resolve_tournament_view_target,
            serve_tournament_view,
        )

        if args.list_runs:
            print(
                format_tournament_view_runs(
                    list_tournament_view_runs(log_root=args.log_root),
                    log_root=args.log_root,
                )
            )
            return 0

        resolved_target = resolve_tournament_view_target(
            args.target,
            latest=args.latest,
            run_label=args.label,
            job_id=args.job_id,
            log_root=args.log_root,
        )
        if isinstance(resolved_target, (str, bytes)):
            print(f"Using tournament target: {resolved_target}")
        else:
            print(f"Using tournament target: {resolved_target.as_posix()}")
        return serve_tournament_view(
            resolved_target,
            host=args.host,
            port=args.port,
        )

    parser.print_help()
    return 1


def _emit_result(
    result: ResultT,
    *,
    payload_fn: Callable[[ResultT], dict[str, Any]],
    formatter_fn: Callable[[ResultT], str],
    json_out: str | None,
) -> int:
    from ._cli_format import write_json_output

    payload = payload_fn(result)
    output_path = write_json_output(payload, json_out)
    print(formatter_fn(result))
    if output_path is not None:
        print(f"\nSaved JSON output to {output_path.as_posix()}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dfm_evals.tournament.cli")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize tournament state")
    init_parser.add_argument("--config", required=True, help="Path to tournament config")

    generate_parser = subparsers.add_parser(
        "generate", help="Generate contestant responses"
    )
    generate_parser.add_argument(
        "--config", required=True, help="Path to tournament config"
    )
    generate_parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional subset of contestant models",
    )
    generate_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    run_parser = subparsers.add_parser("run", help="Run tournament from config")
    run_parser.add_argument("--config", required=True, help="Path to tournament config")
    run_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )
    run_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    resume_parser = subparsers.add_parser(
        "resume", help="Resume tournament from config path or run dir"
    )
    resume_parser.add_argument("target", help="Config file path, run dir, or state dir path")
    resume_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )
    resume_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    add_model_parser = subparsers.add_parser(
        "add-model", help="Add one or more models to an existing tournament"
    )
    add_model_parser.add_argument(
        "target",
        help="Tournament run directory path (or config/run/state target)",
    )
    add_model_parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Model name to add (repeat for multiple models)",
    )
    add_model_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max batches to execute before returning",
    )
    add_model_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    register_model_parser = subparsers.add_parser(
        "register-model",
        help="Register one or more models without generation or judging",
    )
    register_model_parser.add_argument(
        "target",
        help="Tournament run directory path (or config/run/state target)",
    )
    register_model_parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Model name to register (repeat for multiple models)",
    )
    register_model_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    update_config_parser = subparsers.add_parser(
        "update-config",
        help="Update selected persisted config fields for an existing tournament run",
    )
    update_config_parser.add_argument(
        "target",
        help="Tournament run directory path (or config/run/state target)",
    )
    update_config_parser.add_argument(
        "--max-total-matches",
        type=int,
        default=None,
        help="Persist a new max_total_matches value for this run",
    )
    update_config_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    status_parser = subparsers.add_parser("status", help="Show tournament status")
    status_parser.add_argument("target", help="Config file path, run dir, or state dir path")
    status_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    export_parser = subparsers.add_parser("export", help="Export rankings artifacts")
    export_parser.add_argument("target", help="Config file path, run dir, or state dir path")
    export_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for export files",
    )
    export_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    export_prompts_parser = subparsers.add_parser(
        "export-prompts",
        help="Export prompts with current responses and no judge outputs",
    )
    export_prompts_parser.add_argument(
        "target",
        help="Config file path, run dir, or state dir path",
    )
    export_prompts_parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path (default: <run>/exports/prompt_responses.json)",
    )
    export_prompts_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    export_html_parser = subparsers.add_parser(
        "export-html",
        help="Export a standalone tournament viewer HTML file",
    )
    export_html_parser.add_argument(
        "target",
        help="Config file path, run dir, or state dir path",
    )
    export_html_parser.add_argument(
        "--output",
        default=None,
        help="Optional output HTML path (default: <run>/exports/viewer.html)",
    )
    export_html_parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON output",
    )

    view_parser = subparsers.add_parser(
        "view",
        help="Serve a hosted tournament viewer on a local HTTP port",
    )
    view_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "Optional config/state path, run label, or job id. "
            "If omitted, the latest tournament run under the log root is used."
        ),
    )
    view_group = view_parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--latest",
        action="store_true",
        help="Open the latest tournament run under the log root",
    )
    view_group.add_argument(
        "--label",
        default=None,
        help="Open a tournament run by run label under the log root",
    )
    view_group.add_argument(
        "--job-id",
        default=None,
        help="Open a tournament run by Slurm job id suffix",
    )
    view_parser.add_argument(
        "--log-root",
        default="logs/evals-logs",
        help="Tournament run root for --latest/--label/--job-id lookup (default: logs/evals-logs)",
    )
    view_parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List discovered tournament runs under the log root and exit",
    )
    view_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind host (default: 127.0.0.1)",
    )
    view_parser.add_argument(
        "--port",
        type=int,
        default=7576,
        help="HTTP bind port (default: 7576)",
    )

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
