import contextlib
import datetime
import hashlib
import importlib.abc
import importlib.util
import json
import logging
import os
import sys
from importlib.machinery import ModuleSpec
from optparse import Values
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple
from unittest import mock

import pytest

from pip._vendor.packaging.version import Version

from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import (
    IndexGroupCommand,
    RequirementCommand,
    SessionCommandMixin,
)
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.commands import commands_dict, create_command
from pip._internal.self_outdated_check import UpgradePrompt

# These are the expected names of the commands whose classes inherit from
# IndexGroupCommand.
EXPECTED_INDEX_GROUP_COMMANDS = ["download", "index", "install", "list", "wheel"]


def check_commands(pred: Callable[[Command], bool], expected: List[str]) -> None:
    """
    Check the commands satisfying a predicate.
    """
    commands = [create_command(name) for name in sorted(commands_dict)]
    actual = [command.name for command in commands if pred(command)]
    assert actual == expected, f"actual: {actual}"


def test_commands_dict__order() -> None:
    """
    Check the ordering of commands_dict.
    """
    names = list(commands_dict)
    # A spot-check is sufficient to check that commands_dict encodes an
    # ordering.
    assert names[0] == "install"
    assert names[-1] == "help"


@pytest.mark.parametrize("name", list(commands_dict))
def test_create_command(name: str) -> None:
    """Test creating an instance of each available command."""
    command = create_command(name)
    assert command.name == name
    assert command.summary == commands_dict[name].summary


def test_session_commands() -> None:
    """
    Test which commands inherit from SessionCommandMixin.
    """

    def is_session_command(command: Command) -> bool:
        return isinstance(command, SessionCommandMixin)

    expected = ["download", "index", "install", "list", "search", "uninstall", "wheel"]
    check_commands(is_session_command, expected)


def test_index_group_commands() -> None:
    """
    Test the commands inheriting from IndexGroupCommand.
    """

    def is_index_group_command(command: Command) -> bool:
        return isinstance(command, IndexGroupCommand)

    check_commands(is_index_group_command, EXPECTED_INDEX_GROUP_COMMANDS)

    # Also check that the commands inheriting from IndexGroupCommand are
    # exactly the commands with the --no-index option.
    def has_option_no_index(command: Command) -> bool:
        return command.parser.has_option("--no-index")

    check_commands(has_option_no_index, EXPECTED_INDEX_GROUP_COMMANDS)


@pytest.mark.parametrize("command_name", EXPECTED_INDEX_GROUP_COMMANDS)
@pytest.mark.parametrize(
    "disable_pip_version_check, no_index, expected_called",
    [
        # The fetch phase only runs when both disable_pip_version_check
        # and no_index are False.
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (True, True, False),
    ],
)
@mock.patch("pip._internal.cli.index_command._pip_self_version_check_fetch")
def test_index_group_pip_version_check(
    mock_version_check: mock.Mock,
    command_name: str,
    disable_pip_version_check: bool,
    no_index: bool,
    expected_called: bool,
) -> None:
    """
    Test whether the pre-body fetch runs when ``pip_version_check()`` is
    entered, for each of the IndexGroupCommand classes.
    """
    command = create_command(command_name)
    options = command.parser.get_default_values()
    options.disable_pip_version_check = disable_pip_version_check
    options.no_index = no_index
    # Return None so the emit branch is a no-op.
    mock_version_check.return_value = None

    # See test test_list_pip_version_check() below.
    if command_name == "list":
        expected_called = False

    with command.pip_version_check(options, []):
        pass
    if expected_called:
        mock_version_check.assert_called_once()
    else:
        mock_version_check.assert_not_called()


@mock.patch("pip._internal.cli.index_command._pip_self_version_check_fetch")
def test_install_pip_version_check_skipped_when_pip_is_a_requirement(
    mock_version_check: mock.Mock,
) -> None:
    """``pip install pip`` must skip the self-version check: the running pip
    may be replaced before emit."""
    command = create_command("install")
    options = command.parser.get_default_values()
    options.disable_pip_version_check = False
    options.no_index = False

    with command.pip_version_check(options, ["pip"]):
        pass
    mock_version_check.assert_not_called()

    with command.pip_version_check(options, ["pip==25.0"]):
        pass
    mock_version_check.assert_not_called()

    with command.pip_version_check(options, ["some-other-pkg"]):
        pass
    mock_version_check.assert_called_once()


def test_requirement_commands() -> None:
    """
    Test which commands inherit from RequirementCommand.
    """

    def is_requirement_command(command: Command) -> bool:
        return isinstance(command, RequirementCommand)

    check_commands(is_requirement_command, ["download", "install", "wheel"])


@pytest.mark.parametrize("flag", ["", "--outdated", "--uptodate"])
@mock.patch("pip._internal.cli.index_command._pip_self_version_check_fetch")
@mock.patch.dict(os.environ, {"PIP_DISABLE_PIP_VERSION_CHECK": "no"})
def test_list_pip_version_check(version_check_mock: mock.Mock, flag: str) -> None:
    """
    Ensure that pip list doesn't perform a version self-check unless given
    --outdated or --uptodate (as they require hitting the network anyway).
    """
    command = create_command("list")
    command.run = lambda *args, **kwargs: 0  # type: ignore[method-assign]
    command.main([flag])
    if flag != "":
        version_check_mock.assert_called_once()
    else:
        version_check_mock.assert_not_called()


def _make_version_checking_command(
    command_name: str, cache_dir: Path
) -> Tuple[Command, Values]:
    command = create_command(command_name)
    options = command.parser.get_default_values()
    options.disable_pip_version_check = False
    options.no_index = False
    options.cache_dir = str(cache_dir)
    return command, options


@pytest.mark.parametrize("body_raises", [False, True])
def test_index_group_pip_version_check_fetches_before_body_and_emits_after(
    body_raises: bool, tmp_path: Path
) -> None:
    """
    The self version check must gather everything it needs before the
    command body runs, and only emit the notice once the body has finished
    (even when the body fails).
    """
    events: List[str] = []
    prompt = UpgradePrompt(old="1.0", new="2.0")

    def fake_fetch(session: object, options: Values) -> UpgradePrompt:
        events.append("fetch")
        return prompt

    def fake_emit(upgrade_prompt: Optional[UpgradePrompt]) -> None:
        events.append(f"emit {upgrade_prompt}")

    command, options = _make_version_checking_command("install", tmp_path)
    with mock.patch(
        "pip._internal.cli.index_command._pip_self_version_check_fetch",
        side_effect=fake_fetch,
    ), mock.patch(
        "pip._internal.cli.index_command._pip_self_version_check_emit",
        side_effect=fake_emit,
    ):
        if body_raises:
            with pytest.raises(RuntimeError):
                with command.pip_version_check(options, []):
                    events.append("body")
                    raise RuntimeError("body failed")
        else:
            with command.pip_version_check(options, []):
                events.append("body")

    assert events == ["fetch", "body", f"emit {prompt}"]


@mock.patch("pip._internal.cli.index_command._pip_self_version_check_emit")
def test_index_group_pip_version_check_emits_nothing_when_fetch_fails(
    mock_emit: mock.Mock, tmp_path: Path
) -> None:
    """
    When the pre-body fetch fails, the error is swallowed and there is no
    upgrade prompt to show after the command body ran.
    """
    command, options = _make_version_checking_command("install", tmp_path)
    with mock.patch(
        "pip._internal.cli.index_command._pip_self_version_check_fetch",
        side_effect=RuntimeError("no network"),
    ):
        with command.pip_version_check(options, []):
            pass
    mock_emit.assert_called_once_with(None)


_SELF_CHECK_MODULE = "pip._internal.self_outdated_check"
_INSTALL_COMMAND_MODULE = "pip._internal.commands.install"


class _AttackerLoader(importlib.abc.Loader):
    """Stands in for a malicious copy of pip's own module, written to disk
    by a wheel that ``pip install`` has just installed."""

    def __init__(self, events: List[str]) -> None:
        self.events = events

    def create_module(self, spec: ModuleSpec) -> Optional[ModuleType]:
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.events.append("attacker code executed")


class _OverwrittenSelfCheckFinder(importlib.abc.MetaPathFinder):
    """Simulate ``self_outdated_check`` being overwritten during the command.

    Imports made before the command body starts resolve to the genuine module.
    Once the body has started (and could have installed wheels overwriting
    pip's files), any fresh import of the module runs attacker code instead.
    """

    def __init__(self) -> None:
        self.events: List[str] = []
        self.body_started = False

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        if fullname != _SELF_CHECK_MODULE:
            return None
        if not self.body_started:
            self.events.append("import")
            # Let the regular finders load the genuine module.
            return None
        self.events.append("import after body")
        return importlib.util.spec_from_loader(fullname, _AttackerLoader(self.events))


@contextlib.contextmanager
def _self_check_overwritten_during_body() -> Iterator[_OverwrittenSelfCheckFinder]:
    """Simulate a fresh ``pip install`` process whose own
    ``self_outdated_check`` module gets overwritten by the command body.

    Neither ``self_outdated_check`` nor the install command module are loaded
    yet when the command is created, as in a new pip process.
    """
    import pip._internal
    import pip._internal.commands

    missing = object()
    saved: List[Tuple[Any, str]] = [
        (sys.modules, _SELF_CHECK_MODULE),
        (pip._internal.__dict__, "self_outdated_check"),
        (sys.modules, _INSTALL_COMMAND_MODULE),
        (pip._internal.commands.__dict__, "install"),
    ]
    saved_values = [namespace.pop(key, missing) for namespace, key in saved]
    finder = _OverwrittenSelfCheckFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield finder
    finally:
        sys.meta_path.remove(finder)
        for (namespace, key), value in zip(saved, saved_values):
            namespace.pop(key, None)
            if value is not missing:
                namespace[key] = value


def _seed_fresh_self_check_state(cache_dir: Path, pypi_version: str) -> None:
    """Write a recent selfcheck state file so no network access is needed."""
    selfcheck_dir = cache_dir / "selfcheck"
    selfcheck_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "key": sys.prefix,
        "last_check": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pypi_version": pypi_version,
    }
    statefile_name = hashlib.sha224(sys.prefix.encode()).hexdigest()
    (selfcheck_dir / statefile_name).write_text(json.dumps(state), encoding="utf-8")


def _run_install_with_version_check(
    finder: _OverwrittenSelfCheckFinder,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: Sequence[str] = (),
    build_session_fails: bool = False,
) -> int:
    command, options = _make_version_checking_command("install", cache_dir)

    def run(options: Values, args: List[str]) -> int:
        # Installing wheels may overwrite pip's own modules from here on.
        finder.events.append("body")
        finder.body_started = True
        return SUCCESS

    monkeypatch.setattr(command, "run", run)
    if build_session_fails:

        def broken_build_session(*args: object, **kwargs: object) -> None:
            raise RuntimeError("cannot build session")

        monkeypatch.setattr(command, "_build_session", broken_build_session)

    return command._run_wrapper(logging.INFO, options, list(args))


@pytest.mark.parametrize("args", [[], ["some-other-pkg"], ["pip"], ["pip==999.0"]])
@pytest.mark.parametrize("pypi_version", ["1.0", "999.0"])
def test_install_self_version_check_module_not_imported_after_body(
    pypi_version: str,
    args: List[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CVE-2026-6357: a wheel installed by ``pip install`` can overwrite pip's own
    ``self_outdated_check`` module. It must be imported before any wheel is
    installed, so the overwritten copy is never executed.
    """
    _seed_fresh_self_check_state(tmp_path, pypi_version)
    with _self_check_overwritten_during_body() as finder:
        status = _run_install_with_version_check(
            finder, tmp_path, monkeypatch, args=args
        )

    assert status == SUCCESS
    assert "attacker code executed" not in finder.events
    assert finder.events == ["import", "body"]


def test_install_self_version_check_failed_fetch_not_imported_after_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    CVE-2026-6357: if the pre-body part of the self version check fails before
    reaching ``self_outdated_check``, the module must still not be imported
    after the command body ran.
    """
    _seed_fresh_self_check_state(tmp_path, "999.0")
    with _self_check_overwritten_during_body() as finder:
        status = _run_install_with_version_check(
            finder, tmp_path, monkeypatch, build_session_fails=True
        )

    assert status == SUCCESS
    assert "attacker code executed" not in finder.events
    assert finder.events == ["import", "body"]


@mock.patch("pip._internal.self_outdated_check._get_current_remote_pip_version")
@mock.patch("pip._internal.self_outdated_check.get_default_environment")
def test_install_self_version_check_reads_environment_before_body(
    mock_get_env: mock.Mock,
    mock_get_remote: mock.Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CVE-2026-6357: the installed pip distribution (version and installer) is
    inspected before the command body runs, never afterwards, when the
    environment may already have been rewritten by installed wheels.
    """
    events: List[str] = []

    def get_default_environment() -> mock.Mock:
        events.append("read environment")
        installed_dist = mock.Mock()
        installed_dist.version = Version("1.0")
        installed_dist.installer = "pip"
        return mock.Mock(get_distribution=mock.Mock(return_value=installed_dist))

    mock_get_env.side_effect = get_default_environment
    mock_get_remote.side_effect = lambda session, options: "2.0"

    command, options = _make_version_checking_command("install", tmp_path)

    def run(options: Values, args: List[str]) -> int:
        events.append("body")
        return SUCCESS

    monkeypatch.setattr(command, "run", run)
    with mock.patch(
        "pip._internal.cli.index_command._pip_self_version_check_emit",
        side_effect=lambda prompt: events.append(f"emit {prompt}"),
    ):
        status = command._run_wrapper(logging.INFO, options, [])

    assert status == SUCCESS
    assert events == [
        "read environment",
        "body",
        f"emit {UpgradePrompt(old='1.0', new='2.0')}",
    ]
