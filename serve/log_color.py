"""serve/log_color.py - color for the server's own console lines, in the palette of
the vLLM fork's ColoredFormatter (vllm/logging_utils/formatter.py), so a Strata
console reads like the vLLM one (issue stack/strata#5, API-PARITY-PLAN section 10):

    grey     the time stamp of each line and the "[strata]" source tag
             (vLLM greys its timestamp and its [file:line] info)
    level    DEBUG white, INFO green, WARNING yellow, ERROR red, CRITICAL magenta
             (vLLM colors the level name of every line; Strata's own lines carry
             the level word bare at the head, after the tag or after an indent)

The lifecycle lines (serve/lifecycle_log.py) carry their own vLLM palette already
and are not touched here - console_line only ever greys their stamp.

Gate (the same rules as vllm/logger.py's _use_color, so the same env names work
across both backends): NO_COLOR forces color off, VLLM_LOGGING_COLOR=1 forces it
on, "0" forces it off, unset follows the console (color only on a TTY). Only the
server's own console output (_ColoredOut in serve/server.py) is colorized: the
engine's log file stays plain, so greps (strata.sh's "ready: http"), the
start-failure hints and the narrator's log reads are unchanged - and with color
on, the greppable text ("ready: http", "[strata]") is only wrapped in escapes at
its head, never inside. View a colored log file with `less -R`.
"""

import os
import re
import sys

# The vLLM ColoredFormatter palette, verbatim (its COLORS dict, GREY and RESET).
LEVEL_COLORS = {
    "DEBUG": "\033[37m",     # white
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
}
GREY = "\033[90m"
RESET = "\033[0m"


def _use_color() -> bool:
    """The vllm/logger.py rule: NO_COLOR or VLLM_LOGGING_COLOR=0 off, =1 on,
    unset only on a TTY. Read once at import, like the vLLM config: a
    deployment-level switch."""
    if os.environ.get("NO_COLOR"):
        return False
    env = os.environ.get("VLLM_LOGGING_COLOR")
    if env == "0":
        return False
    if env == "1":
        return True
    try:                                    # unset: auto-off when not a TTY
        return bool(sys.stdout) and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


_COLOR = _use_color()

# _stamp's stamp at the head of a line ("12:34:56.789 ...").
_STAMP_HEAD = re.compile(r"^(\d\d:\d\d:\d\d\.\d\d\d)( )")

# The source tag, grey like vLLM's [file:line] info ("[strata] ...").
_TAG_HEAD = re.compile(r"^([ \t]*)(\[strata\])( )")

# A level word opens the body: the shapes server.py prints -
# "       WARNING: no API key ...", "[strata] WARNING: RAM is tight ..." and
# (after the tag has been greyed) "\x1b[90m[strata]\x1b[0m WARNING: ...".
_LEVEL_AT_HEAD = re.compile(r"^([ \t]*)(\x1b\[90m)?(\[strata\](?:\x1b\[0m)? )?"
                            r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)(?=[ :(])")


def color_enabled() -> bool:
    return _COLOR


def level(text: str) -> str:
    """The body's leading level word in its vLLM level color; anything else
    untouched (a level word mid-line is text, not a level)."""
    if not _COLOR:
        return text
    m = _LEVEL_AT_HEAD.match(text)
    if not m:
        return text
    pad, grey_on, tag, word = m[1], m[2] or "", m[3] or "", m[4]
    return f"{pad}{grey_on}{tag}{LEVEL_COLORS[word]}{word}{RESET}{text[m.end():]}"


def console_line(text: str) -> str:
    """One console line (stamp included, as _StampedOut has just made it): the
    stamp and the [strata] tag grey, the leading level word in its color. Plain
    text when color is off, so greps and tests are unchanged."""
    if not _COLOR:
        return text
    m = _STAMP_HEAD.match(text)
    body = text[m.end():] if m else text
    t = _TAG_HEAD.match(body)
    if t:                                   # grey the source tag before level()
        body = f"{t[1]}{GREY}{t[2]}{RESET}{t[3]}{body[t.end():]}"
    body = level(body)
    if not m:
        return body
    return f"{GREY}{m[1]}{RESET}{m[2]}{body}"
