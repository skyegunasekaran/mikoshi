"""Handle-relative file operations for untrusted, concurrently changing trees.

POSIX uses openat-style descriptors. Windows uses directory-relative NT handles;
neither backend re-resolves a checked parent path when opening its children.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


class UnsafePathError(ValueError):
    pass


class FileChangedError(ValueError):
    pass


def validate_component(name, *, windows=None):
    """Accept one literal path component, excluding Windows aliases and streams."""
    if not name or name in {".", ".."} or any(character in name for character in "/\\\x00"):
        raise UnsafePathError("file names must be single normalized path components")
    if windows if windows is not None else os.name == "nt":
        reserved = {"con", "prn", "aux", "nul", "conin$", "conout$", "clock$"}
        stem = name.split(".", 1)[0].rstrip(" .").casefold()
        if (
            name[-1] in " ."
            or any(ord(character) < 32 or character in '<>:"|?*' for character in name)
            or stem in reserved
            or (len(stem) == 4 and stem[:3] in {"com", "lpt"} and stem[3] in "123456789¹²³")
            or PureWindowsPath(name).drive
        ):
            raise UnsafePathError(
                "Windows file names must not use drives, streams, devices, or aliases"
            )


def is_reparse(info):
    return bool(getattr(info, "st_file_attributes", 0) & 0x400)


def is_link_or_reparse(path):
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or is_reparse(info)


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    info: object


class PosixDirectory:
    def __init__(self, descriptor):
        self.descriptor = descriptor

    @classmethod
    def anchor(cls, anchor):
        return cls(os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW))

    def close(self):
        os.close(self.descriptor)

    def info(self):
        return os.fstat(self.descriptor)

    def child(self, name, *, create=False, exclusive=False):
        validate_component(name)
        if create:
            try:
                os.mkdir(name, dir_fd=self.descriptor)
            except FileExistsError:
                if exclusive:
                    raise
        try:
            return type(self)(
                os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.descriptor)
            )
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise UnsafePathError(
                    "symlinks and non-directories are not allowed in paths"
                ) from error
            raise

    def open_file(self, name, *, create=False, mode=0o600):
        validate_component(name)
        flags = (
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            if create
            else os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        try:
            return os.open(name, flags, mode, dir_fd=self.descriptor)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise UnsafePathError("symlinks are not allowed in paths") from error
            raise

    def entries(self):
        with os.scandir(self.descriptor) as children:
            for child in children:
                try:
                    yield DirectoryEntry(child.name, child.stat(follow_symlinks=False))
                except FileNotFoundError:
                    continue

    def set_mode(self, mode):
        os.fchmod(self.descriptor, mode)

    @staticmethod
    def file_mode(descriptor, mode):
        os.fchmod(descriptor, mode)

    def _publish(self, descriptor, temporary, target, replace):
        if replace:
            os.replace(temporary, target, src_dir_fd=self.descriptor, dst_dir_fd=self.descriptor)
        else:
            os.link(
                temporary,
                target,
                src_dir_fd=self.descriptor,
                dst_dir_fd=self.descriptor,
                follow_symlinks=False,
            )

    def _discard_stage(self, descriptor, temporary, published):
        try:
            os.unlink(temporary, dir_fd=self.descriptor)
        except FileNotFoundError:
            pass

    def write_file(self, name, data, mode, *, replace, before_publish=lambda: None):
        """Stage, check, then atomically publish within this pinned directory."""
        validate_component(name)
        temporary = f".write-{secrets.token_hex(16)}"
        descriptor = self.open_file(temporary, create=True)
        published = False
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(descriptor)
                self.file_mode(descriptor, mode)
            before_publish()
            self._publish(descriptor, temporary, name, replace)
            published = True
        finally:
            try:
                self._discard_stage(descriptor, temporary, published)
            finally:
                os.close(descriptor)


@contextmanager
def open_directory(path, parts=(), *, create=False):
    """Pin an absolute root and every component traversed below it.

    Holding all ancestor handles also denies Windows rename/delete operations
    until the operation completes. A link substituted before an open is rejected;
    a link substituted after it cannot redirect an already-open directory handle.
    """
    path = Path(os.path.abspath(path))
    if os.name == "nt":
        from mikoshi.windows_filesystem import WindowsDirectory

        backend = WindowsDirectory
    else:
        backend = PosixDirectory
    with ExitStack() as stack:
        directory = backend.anchor(path.anchor)
        stack.callback(directory.close)
        for part in path.parts[1:]:
            directory = directory.child(part)
            stack.callback(directory.close)
        for part in parts:
            directory = directory.child(part, create=create)
            stack.callback(directory.close)
        yield directory


@contextmanager
def regular_file(directory, name):
    descriptor = directory.open_file(name)
    try:
        info = os.fstat(descriptor)
        if info.st_nlink == 0:
            raise FileChangedError("file was removed while opening; retry")
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or is_reparse(info):
            raise UnsafePathError("files must be regular files without links or reparse points")
        yield descriptor
    finally:
        os.close(descriptor)
