#!/usr//bin/env python3

import json
import argparse
import io
import os
import re
import sys  # For stderr
import urllib.parse
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Google Drive API Imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Progress Bar
try:
    from tqdm import tqdm
except ImportError:
    print(
        "Warning: tqdm library not found. Progress bar disabled. Install with 'pip install tqdm' or add to shell.nix.",
        file=sys.stderr,
    )

    def tqdm(iterable, *args, **kwargs):
        if not kwargs.get("disable", False):
            print("Processing...", file=sys.stderr)  # Print to stderr
        return iterable


# --- Configuration ---
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "token.json"
DEFAULT_COLLAPSE_THRESHOLD = 10  # Lines for Model response collapse

# --- Color Helper (unchanged) ---
COLORS = {
    "reset": "\033[0m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "grey": "\033[90m",
}
USE_COLOR = True  # Will be checked against isatty later


def colorize(text: str, color: str) -> str:
    if USE_COLOR and color in COLORS:
        return f"{COLORS[color]}{text}{COLORS['reset']}"
    return text


# --- Helper Functions (Unchanged: format_token_count, sanitize_filename) ---
def format_token_count(count: int) -> str:
    if count < 10000:
        return str(count)
    return f"{count:,}"


def sanitize_filename(filename: str) -> str:
    # Basic sanitization, can be improved
    sanitized = "".join(c for c in filename if ord(c) >= 32)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", sanitized)
    sanitized = sanitized.strip(". ")
    max_len = 240  # Common filesystem limit component
    encoded = sanitized.encode("utf-8")
    if len(encoded) > max_len:
        # Truncate based on bytes, decode ignoring errors
        sanitized = encoded[:max_len].decode("utf-8", errors="ignore")
    if not sanitized:
        sanitized = "_unnamed_file_"
    return sanitized


def verbose_print(message: str, verbose_flag: bool):
    if verbose_flag:
        print(colorize(message, "magenta"), file=sys.stderr)  # Print verbose to stderr


# --- Drive Functions (Unchanged: get_drive_service, download_file) ---
# Note: Download progress prints are removed; tqdm handles overall progress.
# Error/status prints now use colorize and stderr.
def get_drive_service(credentials_path: Path, verbose: bool):
    creds = None
    token_path = credentials_path.parent / TOKEN_FILE
    verbose_print(f"Checking for token file at: {token_path}", verbose)
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            verbose_print("Token file found.", verbose)
        except Exception as e:
            print(
                colorize(
                    f"Warning: Could not load token file {token_path}: {e}", "yellow"
                ),
                file=sys.stderr,
            )
            creds = None

    if not creds or not creds.valid:
        verbose_print("No valid credentials found or token expired.", verbose)
        if creds and creds.expired and creds.refresh_token:
            try:
                verbose_print("Attempting to refresh expired credentials...", verbose)
                creds.refresh(Request())
                verbose_print("Credentials refreshed successfully.", verbose)
            except Exception as e:
                print(
                    colorize(
                        f"Error refreshing token: {e}. Please re-authenticate.", "red"
                    ),
                    file=sys.stderr,
                )
                try:
                    token_path.unlink(missing_ok=True)
                except OSError as unlink_error:
                    print(
                        colorize(
                            f"Warning: Could not delete invalid token file {token_path}: {unlink_error}",
                            "yellow",
                        ),
                        file=sys.stderr,
                    )
                creds = None
        else:
            if not credentials_path.is_file():
                print(
                    colorize(
                        f"Error: Credentials file not found at {credentials_path}. Please download OAuth credentials.",
                        "red",
                    ),
                    file=sys.stderr,
                )
                return None
            try:
                verbose_print(
                    f"Loading client secrets from: {credentials_path}", verbose
                )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), SCOPES
                )
                print(
                    colorize(
                        "\nPlease authenticate via console (copy/paste URL and code)...",
                        "cyan",
                    ),
                    file=sys.stderr,
                )
                creds = flow.run_console()
                print(colorize("Authentication successful.", "green"), file=sys.stderr)
            except Exception as e:
                print(
                    colorize(f"Error during authentication flow: {e}", "red"),
                    file=sys.stderr,
                )
                return None
        try:
            with open(token_path, "w") as token:
                token.write(creds.to_json())
            print(f"Credentials saved to {token_path}", file=sys.stderr)
        except Exception as e:
            print(
                colorize(
                    f"Warning: Could not save token file {token_path}: {e}", "yellow"
                ),
                file=sys.stderr,
            )

    try:
        verbose_print("Building Drive service...", verbose)
        service = build("drive", "v3", credentials=creds)
        verbose_print("Drive service built successfully.", verbose)
        return service
    except HttpError as error:
        print(
            colorize(f"An error occurred building Drive service: {error}", "red"),
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            colorize(
                f"An unexpected error occurred building Drive service: {e}", "red"
            ),
            file=sys.stderr,
        )
        return None


def download_file(
    service, file_id: str, download_dir: Path, force: bool, verbose: bool
) -> Tuple[bool, Optional[Path], Optional[str]]:
    """Downloads a file, returns success, local_path, mime_type."""
    if not service:
        print(
            colorize("Skipping download: Drive service not available.", "yellow"),
            file=sys.stderr,
        )
        return False, None, None
    mime_type = None
    try:
        verbose_print(f"Getting metadata for Drive file ID: {file_id}", verbose)
        file_metadata = (
            service.files().get(fileId=file_id, fields="name, mimeType").execute()
        )
        original_file_name = file_metadata.get("name", file_id)
        mime_type = file_metadata.get("mimeType")
    except HttpError as error:
        print(
            colorize(
                f"\nAn HTTP error occurred getting metadata for {file_id}: {error}",
                "red",
            ),
            file=sys.stderr,
        )
        return False, None, None  # Handle metadata errors
    except Exception as e:
        print(
            colorize(
                f"\nAn unexpected error occurred getting metadata for {file_id}: {e}",
                "red",
            ),
            file=sys.stderr,
        )
        return False, None, None

    safe_original_name = sanitize_filename(original_file_name)
    local_full_path = (download_dir / safe_original_name).resolve()
    verbose_print(f"Target local path: {local_full_path}", verbose)

    if not force and local_full_path.exists():
        print(
            colorize(
                f"\nSkipping download: '{safe_original_name}' already exists (use --force-download)",
                "yellow",
            ),
            file=sys.stderr,
        )
        return True, local_full_path, mime_type

    try:
        print(
            colorize(
                f"\nAttempting to download '{original_file_name}' (ID: {file_id})...",
                "cyan",
            ),
            file=sys.stderr,
        )
        request = service.files().get_media(fileId=file_id)
        download_dir.mkdir(parents=True, exist_ok=True)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        # No progress bar, just indicate activity
        print(colorize("Downloading...", "cyan"), file=sys.stderr, end="\r")
        while done is False:
            status, done = downloader.next_chunk()
        print("           ", file=sys.stderr, end="\r")  # Clear indicator

        verbose_print(f"Writing downloaded data to {local_full_path}", verbose)
        with open(local_full_path, "wb") as f:
            f.write(fh.getvalue())
        print(
            colorize(f"Successfully downloaded to '{local_full_path}'", "green"),
            file=sys.stderr,
        )
        return True, local_full_path, mime_type

    except HttpError as error:
        print(
            colorize(f"\nAn HTTP error occurred downloading {file_id}: {error}", "red"),
            file=sys.stderr,
        )
        if error.resp.status == 403:
            print(
                colorize("Suggestion: Check permissions or Tester status.", "yellow"),
                file=sys.stderr,
            )
        elif error.resp.status == 404:
            print(
                colorize("Suggestion: File may be deleted or ID incorrect.", "yellow"),
                file=sys.stderr,
            )
        return False, None, mime_type
    except Exception as e:
        print(
            colorize(
                f"\nAn unexpected error occurred downloading {file_id}: {e}", "red"
            ),
            file=sys.stderr,
        )
        return False, None, mime_type


def calculate_token_counts_revised(
    chunks: List[Dict[str, Any]],
) -> Tuple[int, int, Dict[int, int]]:
    """Calculates token counts as previously defined."""
    # (Function unchanged)
    total_input_accumulator = 0
    total_output_accumulator = 0
    running_context_sum = 0
    context_sums = {}
    for i, chunk in enumerate(chunks):
        token_count = chunk.get("tokenCount", 0)
        if chunk.get("role") == "model":
            context_sums[i] = running_context_sum
            total_input_accumulator += running_context_sum
            total_output_accumulator += token_count
        running_context_sum += token_count
    return total_input_accumulator, total_output_accumulator, context_sums


# --- Formatting Functions (Revised for New Callout Structure) ---


def format_user_text(chunk: Dict[str, Any]) -> str:
    text = chunk.get("text", "*No text content*").strip()  # Strip whitespace
    tokens = chunk.get("tokenCount", 0)
    tokens_f = format_token_count(tokens)
    title = f"User: ({tokens_f}T)"

    # Indent the main text to be part of the blockquote
    # *** Use predicate to force indentation on blank lines ***
    indented_text = ""
    if text:  # Only indent if there is text
        indented_text = textwrap.indent(
            text, "> ", predicate=lambda line: True  # Apply prefix to ALL lines
        )

    # User messages default expanded (+)
    # Add final newline only if there is indented text
    final_newline = "\n" if indented_text else ""
    return f"> [!QUESTION]+ {title}\n{indented_text}{final_newline}"


def format_user_drive_doc(
    chunk: Dict[str, Any],
    download_success: bool,
    local_path: Optional[Path],
    output_dir: Path,
    mime_type: Optional[str],
) -> str:
    doc_info = chunk.get("driveDocument", {})
    doc_id = doc_info.get("id", "UNKNOWN_ID")
    tokens = chunk.get("tokenCount", 0)
    tokens_f = format_token_count(tokens)
    title = f"User (attachment): ({tokens_f}T)"
    content = ""
    is_embed = False

    if download_success and local_path:
        try:
            relative_local_path_str = os.path.relpath(local_path, start=output_dir)
        except ValueError:
            relative_local_path_str = str(local_path)
        encoded_path = urllib.parse.quote(relative_local_path_str.replace(os.sep, "/"))
        file_name = local_path.name

        is_text_type = local_path.suffix.lower() in [
            ".txt",
            ".md",
            ".py",
            ".sh",
            ".nix",
            ".json",
            ".yaml",
            ".yml",
            ".log",
        ]

        # Always link attachments; do not embed
        content = f"[{file_name}]({encoded_path}) *(Locally downloaded)*"
    else:  # Fallback to remote link
        remote_link = f"https://drive.google.com/file/d/{doc_id}"
        status = (
            colorize("*(Download failed)*", "red")
            if local_path is None and download_success is False
            else ""
        )
        content = f"[Drive Document ID: {doc_id}]({remote_link}) {status}"

    # Always expanded (+) for user attachments
    fold_char = "+"
    indented_content = textwrap.indent(content, "> ")
    return f"> [!QUOTE]{fold_char} {title}\n{indented_content}\n"


def format_user_drive_image(
    chunk: Dict[str, Any],
    download_success: bool,
    local_path: Optional[Path],
    output_dir: Path,
    mime_type: Optional[str],
) -> str:
    img_info = chunk.get("driveImage", {})
    img_id = img_info.get("id", "UNKNOWN_ID")
    tokens = chunk.get("tokenCount", 0)
    tokens_f = format_token_count(tokens)
    title = f"User (image): ({tokens_f}T)"
    content = ""

    if download_success and local_path:
        try:
            relative_local_path_str = os.path.relpath(local_path, start=output_dir)
        except ValueError:
            relative_local_path_str = str(local_path)
        encoded_path = urllib.parse.quote(relative_local_path_str.replace(os.sep, "/"))
        file_name = local_path.name
        # Always link image attachments; do not embed
        content = f"[{file_name}]({encoded_path})\n> *(Locally downloaded)*"
    else:
        remote_link = f"https://drive.google.com/file/d/{img_id}"
        status = (
            colorize("*(Download failed)*", "red")
            if local_path is None and download_success is False
            else ""
        )
        content = f"[Drive Image ID: {img_id}]({remote_link}) {status}"

    # Images default expanded (+)
    indented_content = textwrap.indent(content, "> ")
    return f"> [!TIP]+ {title}\n{indented_content}\n"


# --- REVISED format_model_combined function (Force Indent on Blank Lines) ---
def format_model_combined(
    thought_chunk: Optional[Dict[str, Any]],
    response_chunk: Dict[str, Any],
    context_sum: int,
    collapse_threshold: int,
) -> str:
    thought_block_final_md = ""  # Initialize empty thought block string
    thought_tokens = 0
    if thought_chunk:
        thought_text_raw = thought_chunk.get(
            "text", "*No thought content*"
        ).strip()  # Get raw text, strip whitespace
        if thought_text_raw:  # Only proceed if there is actual thought text
            thought_tokens = thought_chunk.get("tokenCount", 0)
            thought_tokens_f = format_token_count(thought_tokens)
            thought_title = f"Model Thought: ({thought_tokens_f}T)"

            # Build the nested callout header (Starts with '> > ')
            nested_thought_header = f"> > [!QUESTION]- {thought_title}"

            # Indent the raw thought text to be content of the nested callout (Starts with '> > ')
            # *** Use predicate to force indentation on blank lines ***
            indented_thought_text = textwrap.indent(
                thought_text_raw,
                "> > ",
                predicate=lambda line: True,  # Apply prefix to ALL lines
            )

            # Combine the header and the indented text for the final thought block string
            thought_block_final_md = (
                f"{nested_thought_header}\n{indented_thought_text}\n"
            )
        # No explicit warning needed here now, empty thoughts handled by if thought_text_raw

    # --- Response part (logic unchanged, but add predicate here too) ---
    response_text_raw = response_chunk.get("text", "*No response content*").strip()
    response_tokens = response_chunk.get("tokenCount", 0)
    combined_output_tokens = (
        thought_tokens + response_tokens
    )  # thought_tokens might be 0 if thought was empty

    context_f = format_token_count(context_sum)
    output_f = format_token_count(combined_output_tokens)
    finish_reason = response_chunk.get("finishReason", "UNKNOWN")
    branch_parent = response_chunk.get("branchParent")
    title_core = (
        f"Model: (Context: {context_f}T, Output: {output_f}T) Finish: {finish_reason}"
    )
    if branch_parent:
        parent_display = branch_parent.get("displayName", "N/A")
        title_core += f" (Parent: {parent_display})"

    # Determine folding based on response length
    response_lines = response_text_raw.splitlines() if response_text_raw else []
    is_long_response = (
        collapse_threshold > 0 and len(response_lines) > collapse_threshold
    )
    fold_char = "-" if is_long_response else "+"

    # Indent response text relative to main callout '>' (ONE level deep)
    # *** Use predicate to force indentation on blank lines ***
    indented_response = ""
    if response_text_raw:
        indented_response = textwrap.indent(
            response_text_raw,
            "> ",
            predicate=lambda line: True,  # Apply prefix to ALL lines
        )

    # --- Assemble the final block ---
    callout_header = f"> [!INFO]{fold_char} {title_core}\n"
    # Separator line needs ONE '>' if thought exists AND response exists
    # This provides the desired single blockquoted blank line between sections.
    thought_separator = "> \n" if thought_block_final_md and indented_response else ""

    # Combine, ensuring thought block is placed correctly before the response text
    # Add a final newline after the response block if it exists
    final_newline = "\n" if indented_response else ""
    return f"{callout_header}{thought_block_final_md}{thought_separator}{indented_response}{final_newline}"


# --- Main Processing Function (process_single_file, unchanged structure, calls updated formatters) ---
def process_single_file(
    input_file: Path,
    output_file: Path,
    download_dir: Path,
    credentials_path: Path,
    args: argparse.Namespace,
    prefix_downloads: bool,  # prefix_downloads argument seems unused?
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    global USE_COLOR
    if args.no_color or not os.isatty(
        sys.stderr.fileno()
    ):  # Check stderr for color support
        USE_COLOR = False
    print(
        colorize(f"\n--- Processing File: {input_file.name} ---", "blue"),
        file=sys.stderr,
    )

    if not input_file.is_file():
        print(
            colorize(f"Error: Input file not found: {input_file}", "red"),
            file=sys.stderr,
        )
        return 0, 0, 0, 0, 1, 0, 0, 0, 0
    if input_file == output_file:
        print(
            colorize(f"Error: Input/output collision: {input_file}. Skipping.", "red"),
            file=sys.stderr,
        )
        return 0, 0, 0, 0, 1, 0, 0, 0, 0

    can_download_this_run = (not args.dry_run) and (not args.remote_links)
    drive_service = None
    if can_download_this_run:
        verbose_print(
            "Attempting Google Drive authentication (if needed)...", args.verbose
        )
        drive_service = get_drive_service(credentials_path, args.verbose)
        if not drive_service:
            print(
                colorize("Warning: Drive Auth Failed. Downloads disabled.", "yellow"),
                file=sys.stderr,
            )
    elif args.dry_run:
        verbose_print("Dry Run: Skipping Drive authentication.", args.verbose)
    elif args.remote_links:
        verbose_print("Remote links mode: Skipping Drive authentication.", args.verbose)

    verbose_print(f"Loading JSON data from {input_file}...", args.verbose)
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        verbose_print("JSON data loaded.", args.verbose)
    except Exception as e:
        print(colorize(f"Error reading {input_file}: {e}", "red"), file=sys.stderr)
        return 0, 0, 0, 0, 1, 0, 0, 0, 0

    verbose_print("Calculating token counts...", args.verbose)
    run_settings = data.get("runSettings", {})
    citations = data.get("citations", [])
    chunks = data.get("chunkedPrompt", {}).get("chunks", [])
    if not chunks:
        print(
            colorize(f"Warning: No chunks found in {input_file.name}.", "yellow"),
            file=sys.stderr,
        )
    total_input_tokens, total_output_tokens, context_sums = (
        calculate_token_counts_revised(chunks)
    )
    total_tokens = total_input_tokens + total_output_tokens
    verbose_print(
        f"Tokens: In={total_input_tokens}, Out={total_output_tokens}, Total={total_tokens}",
        args.verbose,
    )

    verbose_print("Generating YAML frontmatter...", args.verbose)
    yaml_frontmatter = "---\n"
    yaml_frontmatter += f"source_json: {input_file.as_posix()}\n"
    # ... (YAML generation as before) ...
    yaml_frontmatter += f"model: \"{run_settings.get('model', 'N/A')}\"\n"
    yaml_frontmatter += f"temperature: {run_settings.get('temperature', 'N/A')}\n"
    yaml_frontmatter += f"topP: {run_settings.get('topP', 'N/A')}\n"
    yaml_frontmatter += f"topK: {run_settings.get('topK', 'N/A')}\n"
    yaml_frontmatter += f"totalInputTokens: {format_token_count(total_input_tokens)}\n"
    yaml_frontmatter += (
        f"totalOutputTokens: {format_token_count(total_output_tokens)}\n"
    )
    yaml_frontmatter += f"totalTokens: {format_token_count(total_tokens)}\n"
    if args.add_tags:
        tags = [tag.strip() for tag in args.add_tags.split(",")]
        if tags:
            yaml_frontmatter += "tags:\n"
            for tag in tags:
                if tag:
                    yaml_frontmatter += f"  - {tag}\n"  # Corrected assignment loop
    if citations:
        yaml_frontmatter += "citations:\n"
        for citation in citations:
            yaml_frontmatter += (
                f"  - \"{citation.get('uri', 'N/A')}\"\n"  # Corrected assignment loop
            )
    yaml_frontmatter += "---\n\n"

    verbose_print("Processing chunks and generating Markdown...", args.verbose)
    markdown_parts = [yaml_frontmatter]
    downloaded_files_summary = []
    failed_downloads = []
    attachments_attempted_count = 0
    num_user_turns = 0
    num_model_turns = 0
    num_thought_blocks = 0
    sum_model_resp_tokens = 0
    i = 0
    output_dir = output_file.parent
    # input_stem = input_file.stem # input_stem not used below

    # Progress bar setup
    pbar_disabled = args.verbose or len(chunks) < 10 or not USE_COLOR
    pbar = tqdm(
        total=len(chunks),
        desc=colorize(f"Processing {input_file.name}", "cyan"),
        unit="chunk",
        disable=pbar_disabled,
        file=sys.stdout,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )  # Ensure tqdm uses stdout

    while i < len(chunks):
        chunk = chunks[i]
        verbose_print(
            f"\nProcessing chunk {i+1}/{len(chunks)}, Role: {chunk.get('role')}",
            args.verbose,
        )
        role = chunk.get("role")
        formatted_output = ""
        download_success = False
        local_path = None
        mime_type = None
        processed_indices = 1
        is_attachment = False
        file_id_to_process = None

        if role == "user":
            num_user_turns += 1
            doc_id = chunk.get("driveDocument", {}).get("id")
            img_id = chunk.get("driveImage", {}).get("id")
            file_id_to_process = doc_id or img_id
            is_attachment = file_id_to_process is not None

        if is_attachment and not args.dry_run:
            if drive_service:
                attachments_attempted_count += 1
                verbose_print(
                    f"Attempting download/metadata for file ID: {file_id_to_process}",
                    args.verbose,
                )
                download_success, local_path, mime_type = download_file(
                    drive_service,
                    file_id_to_process,
                    download_dir,
                    args.force_download,
                    args.verbose,
                )
                if download_success and local_path:
                    downloaded_files_summary.append(local_path)
                elif not download_success:
                    local_path = None
                    failed_downloads.append(file_id_to_process)
            elif not drive_service and not args.remote_links:
                verbose_print(
                    f"Skipping download for {file_id_to_process}: Drive Service unavailable.",
                    args.verbose,
                )
                failed_downloads.append(
                    file_id_to_process
                )  # Count as failed if service failed
            elif args.remote_links:
                verbose_print(
                    f"Skipping download for {file_id_to_process} due to --remote-links.",
                    args.verbose,
                )
        elif args.dry_run and is_attachment:
            verbose_print(
                f"Dry Run: Would attempt download for {file_id_to_process}",
                args.verbose,
            )
            attachments_attempted_count += 1

        if role == "user":
            if "text" in chunk:
                formatted_output = format_user_text(chunk)
            elif "driveDocument" in chunk:
                formatted_output = format_user_drive_doc(
                    chunk,
                    download_success,
                    local_path,
                    output_dir,
                    mime_type,
                )
            elif "driveImage" in chunk:
                formatted_output = format_user_drive_image(
                    chunk, download_success, local_path, output_dir, mime_type
                )
            else:
                formatted_output = f"> [!WARNING]+ Unknown User Chunk Type\n> ```json\n> {textwrap.indent(json.dumps(chunk, indent=2), '> ')}\n> ```\n"  # Indent JSON
            markdown_parts.append(formatted_output)

        elif role == "model":
            num_model_turns += 1
            context_sum = context_sums.get(i, 0)
            is_thought = chunk.get("isThought", False)
            thought_chunk = None
            response_chunk = None
            is_grouped = False
            if is_thought:
                num_thought_blocks += 1
            # --- Grouping Logic ---
            if is_thought and (i + 1) < len(chunks):
                next_chunk = chunks[i + 1]
                # Check if next chunk is a model response (not another thought)
                if next_chunk.get("role") == "model" and not next_chunk.get(
                    "isThought", False
                ):
                    verbose_print("Found thought-response pair.", args.verbose)
                    thought_chunk = chunk
                    response_chunk = next_chunk
                    sum_model_resp_tokens += response_chunk.get(
                        "tokenCount", 0
                    )  # Count tokens of the actual response part
                    # Pass both to the formatter
                    formatted_output = format_model_combined(
                        thought_chunk,
                        response_chunk,
                        context_sum,
                        args.collapse_threshold,
                    )
                    processed_indices = 2  # Processed two chunks (thought + response)
                    is_grouped = True
            # --- Handle non-grouped cases ---
            if not is_grouped:
                if is_thought:  # Standalone thought (no subsequent response found)
                    verbose_print("Found standalone thought chunk.", args.verbose)
                    # Pass thought chunk, but empty response chunk
                    formatted_output = format_model_combined(
                        chunk,
                        {"text": "", "tokenCount": 0, "finishReason": "N/A"},
                        context_sum,
                        args.collapse_threshold,
                    )
                    # processed_indices remains 1
                else:  # Standalone response (not preceded by a thought)
                    verbose_print("Found standalone response chunk.", args.verbose)
                    response_chunk = chunk
                    sum_model_resp_tokens += response_chunk.get("tokenCount", 0)
                    # Pass empty thought chunk, and the actual response chunk
                    formatted_output = format_model_combined(
                        None, response_chunk, context_sum, args.collapse_threshold
                    )
                    # processed_indices remains 1
            markdown_parts.append(formatted_output)
        else:  # Unknown role
            formatted_output = f"> [!ERROR]+ Unknown Chunk Role: {role}\n> ```json\n> {textwrap.indent(json.dumps(chunk, indent=2), '> ')}\n> ```\n"  # Indent JSON
            markdown_parts.append(formatted_output)

        if args.add_separators:
            markdown_parts.append("---\n\n")
        else:
            markdown_parts.append(
                "\n"
            )  # Still add a newline for separation between blocks

        pbar.update(processed_indices)
        i += processed_indices  # Increment by 1 or 2 depending on grouping

    pbar.close()

    # --- Append Attachment Summary ---
    if args.attachment_summary and (downloaded_files_summary or failed_downloads):
        verbose_print("Appending attachment summary...", args.verbose)
        summary_header = "\n\n---\n## Attachments Summary\n"
        summary_list = []
        md_links = []
        if downloaded_files_summary:
            summary_list.append("**Successfully Downloaded:**\n")
            for path in downloaded_files_summary:
                try:
                    relative_path = os.path.relpath(path, start=output_dir)
                except ValueError:
                    relative_path = str(
                        path
                    )  # Handle case where paths are on different drives (Windows)
                encoded_path = urllib.parse.quote(
                    relative_path.replace(os.sep, "/")
                )  # Use forward slashes for Markdown/URL
                md_links.append(f"- [{path.name}]({encoded_path})")
            summary_list.extend(sorted(md_links))  # Sort alphabetically
        if failed_downloads:
            summary_list.append("\n**Failed/Skipped Downloads (IDs):**\n")
            for failed_id in sorted(list(set(failed_downloads))):  # Unique sorted IDs
                remote_link = f"https://drive.google.com/file/d/{failed_id}"
                summary_list.append(f"- [{failed_id}]({remote_link})")
        markdown_parts.append(summary_header + "\n".join(summary_list) + "\n")

    # --- Write Output File ---
    final_markdown = "".join(markdown_parts).strip() + "\n"  # Ensure trailing newline
    if args.dry_run:
        print(
            colorize("\n--- Dry Run: Output Start (Preview) ---", "grey"),
            file=sys.stderr,
        )
        preview = final_markdown[:1500] + ("..." if len(final_markdown) > 1500 else "")
        print(preview, file=sys.stderr)  # Print preview to stderr
        print(colorize("--- Dry Run: Output End ---", "grey"), file=sys.stderr)
        print(
            colorize(
                f"\nDry Run complete for {input_file.name}. No files written or downloaded.",
                "yellow",
            )
        )
    else:
        verbose_print(f"Writing final Markdown to {output_file}...", args.verbose)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_markdown)
            print(
                colorize(
                    f"Successfully converted '{input_file.name}' to '{output_file.name}'",
                    "green",
                )
            )
            if not args.remote_links:
                num_failed = len(set(failed_downloads))  # Count unique failed IDs
                if num_failed > 0:
                    print(
                        colorize(
                            f"Note: {num_failed} attachment download(s) failed/skipped for this file.",
                            "yellow",
                        ),
                        file=sys.stderr,
                    )
        except Exception as e:
            print(
                colorize(f"Error writing file {output_file}: {e}", "red"),
                file=sys.stderr,
            )
            # Return stats even on write error, marking error=1
            return (
                len(chunks),
                attachments_attempted_count,
                len(downloaded_files_summary),
                len(set(failed_downloads)),
                1,
                num_user_turns,
                num_model_turns,
                num_thought_blocks,
                sum_model_resp_tokens,
            )

    # Return statistics for this file
    return (
        len(chunks),
        attachments_attempted_count,
        len(downloaded_files_summary),
        len(set(failed_downloads)),
        0,
        num_user_turns,
        num_model_turns,
        num_thought_blocks,
        sum_model_resp_tokens,
    )


# --- Main Execution Logic (Multiple Files) ---
def main():
    # ArgumentParser setup remains the same as previous version
    parser = argparse.ArgumentParser(
        description="Convert chat log JSON (containing text, Google Drive links, and model thoughts/responses) to Markdown format, suitable for Obsidian.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input/Output Options
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="One or more input JSON chat log file paths.",
    )
    io_group.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save output Markdown file(s). Default: alongside input file(s).",
    )

    # Attachment Download Options
    dl_group = parser.add_argument_group("Attachment Download Options (Google Drive)")
    dl_group.add_argument(
        "--no-download",
        action="store_true",
        help="Disable downloading attachments (enabled by default).",
    )
    dl_group.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Base directory for ALL attachments. Default: Create '{output_stem}_attachments' folder next to each output MD file.",
    )
    dl_group.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download attachments even if local files exist.",
    )
    dl_group.add_argument(
        "--credentials",
        type=Path,
        default="credentials.json",
        help="Path to Google OAuth credentials.json file.",
    )

    # Markdown Formatting Options
    fmt_group = parser.add_argument_group("Markdown Formatting Options")
    fmt_group.add_argument(
        "--add-tags",
        type=str,
        default=None,
        help="Comma-separated tags for YAML frontmatter (e.g., 'chat,ai,project-x').",
    )
    fmt_group.add_argument(
        "--attachment-summary",
        action="store_true",
        help="Append summary of downloaded/failed attachments at the end.",
    )
    fmt_group.add_argument(
        "--remote-links",
        action="store_true",
        help="Do not download attachments; link to Drive URLs instead.",
    )
    fmt_group.add_argument(
        "--add-separators",
        action="store_true",
        help="Add horizontal rule '---' between conversation turns.",
    )
    fmt_group.add_argument(
        "--collapse-threshold",
        type=int,
        default=DEFAULT_COLLAPSE_THRESHOLD,
        help=f"Lines after which model responses are collapsed by default. Set to 0 to disable collapsing.",
    )

    # Script Behavior Options
    beh_group = parser.add_argument_group("Script Behavior Options")
    beh_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate processing without writing files or downloading attachments.",
    )
    beh_group.add_argument(
        "--no-color", action="store_true", help="Disable colorized terminal output."
    )
    beh_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed processing information (disables main progress bar).",
    )

    args = parser.parse_args()

    # No embedding: attachments are always linked. --remote-links skips downloads.

    # --- Setup and Initialization ---
    global USE_COLOR
    # Check stderr for TTY support for colorizing messages
    if args.no_color or not sys.stderr.isatty():
        USE_COLOR = False

    # Initialize counters for the final summary
    total_files_processed = 0
    total_chunks_processed = 0
    total_attachments_attempted = 0
    total_downloads_succeeded = 0
    total_downloads_failed = 0  # Tracks unique failed IDs across all files
    total_errors = 0  # Tracks file-level processing errors
    total_user_turns = 0
    total_model_turns = 0
    total_thought_blocks = 0
    total_model_resp_tokens = 0

    # Resolve paths early
    args.credentials = args.credentials.resolve()
    base_output_dir = args.output_dir.resolve() if args.output_dir else None
    global_download_dir = args.download_dir.resolve() if args.download_dir else None

    print(
        colorize(f"Starting processing for {len(args.input_files)} file(s)...", "blue"),
        file=sys.stderr,
    )
    print(
        colorize(
            f"Output Base Directory: {base_output_dir if base_output_dir else 'Alongside Input'}",
            "cyan",
        ),
        file=sys.stderr,
    )
    if args.remote_links:
        print(colorize("Downloads: SKIPPED (remote-links)", "cyan"), file=sys.stderr)
    else:
        print(colorize("Downloads: ENABLED", "cyan"), file=sys.stderr)
    if not args.remote_links:
        print(
            colorize(f"Credentials Path: {args.credentials}", "cyan"), file=sys.stderr
        )
        print(
            colorize(
                f"Global Download Dir: {global_download_dir if global_download_dir else 'Per-File Attachment Folder'}",
                "cyan",
            ),
            file=sys.stderr,
        )
        if args.force_download:
            print(colorize("Force Download: ENABLED", "yellow"), file=sys.stderr)
    if args.dry_run:
        print(colorize("Dry Run Mode: ENABLED", "yellow"), file=sys.stderr)

    # --- File Processing Loop ---
    # No top-level tqdm; progress is shown per file inside process_single_file
    for input_file_rel in args.input_files:
        input_file = input_file_rel.resolve()
        input_stem = input_file.stem  # Used for default output naming

        # Determine output file path
        current_output_dir = base_output_dir if base_output_dir else input_file.parent
        # Avoid overwriting if input is already .md (append .output.md)
        if input_file.suffix.lower() == ".md":
            output_file = current_output_dir / f"{input_stem}.output.md"
        else:
            output_file = current_output_dir / f"{input_stem}.md"

        # Determine attachment download directory for this specific file
        current_download_dir = global_download_dir
        if current_download_dir is None:  # Default: relative to output file
            output_stem = output_file.stem  # Use the stem of the *output* file
            current_download_dir = (
                output_file.parent / f"{output_stem}_attachments"
            )  # Default folder name

        # Process the file
        (
            chunks,
            attempted,
            succeeded,
            failed,
            errors,
            user_turns,
            model_turns,
            thought_blocks,
            resp_tokens,
        ) = process_single_file(
            input_file,
            output_file,
            current_download_dir,
            args.credentials,
            args,
            False,  # Pass args object, prefixing unused
        )

        # Accumulate statistics
        if errors == 0:
            total_files_processed += 1
        total_chunks_processed += chunks
        total_attachments_attempted += attempted
        total_downloads_succeeded += succeeded
        total_downloads_failed += failed  # Sum of unique failures per file
        total_errors += errors
        total_user_turns += user_turns
        total_model_turns += model_turns
        total_thought_blocks += thought_blocks
        total_model_resp_tokens += resp_tokens

    # --- Final Summary (prints to standard output) ---
    print(f"\n--- Overall Summary ---")
    print(
        f"Files Successfully Processed: {total_files_processed} / {len(args.input_files)}"
    )
    if total_errors > 0:
        print(colorize(f"File Processing Errors Encountered: {total_errors}", "red"))
    print(f"Total Chunks Processed: {format_token_count(total_chunks_processed)}")
    print(f"Total User Turns: {total_user_turns}")
    print(f"Total Model Turns (incl. standalone thoughts): {total_model_turns}")
    print(f"  (Blocks Containing Thoughts: {total_thought_blocks})")

    # Calculate average response tokens (excluding thoughts)
    # Ensure we don't divide by zero if only thoughts occurred
    num_actual_responses = total_model_turns  # Assumes each model turn has *some* response, even if empty with a thought
    if (
        total_thought_blocks == total_model_turns and total_model_turns > 0
    ):  # Edge case: only thoughts, no response parts
        avg_resp_token = 0  # Avoid division by zero, avg response is 0
    elif num_actual_responses > 0:
        avg_resp_token = total_model_resp_tokens / num_actual_responses
    else:
        avg_resp_token = 0
    print(
        f"Avg. Output Tokens per Model Response (excl. thoughts): ~{format_token_count(int(avg_resp_token))}"
    )

    if (
        total_attachments_attempted > 0 or total_downloads_failed > 0
    ):  # Show download stats if anything was attempted or failed
        print(f"\nTotal Attachments Attempted: {total_attachments_attempted}")
        print(
            colorize(
                f"  Total Downloads Succeeded: {total_downloads_succeeded}", "green"
            )
        )
        if total_downloads_failed > 0:
            print(
                colorize(
                    f"  Total Downloads Failed/Skipped (Unique IDs): {total_downloads_failed}",
                    "red",
                )
            )
    print(f"-----------------------")


if __name__ == "__main__":
    main()
