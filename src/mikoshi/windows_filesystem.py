"""Windows handle-relative counterpart to POSIX directory descriptors.

NtCreateFile's RootDirectory and FILE_OPEN_REPARSE_POINT avoid path-based
check/open races. Each open is exactly one component, and reparse attributes are
checked on that handle. Ancestors are retained without FILE_SHARE_DELETE.

API contracts: learn.microsoft.com/windows/win32/api/winternl/nf-winternl-ntcreatefile
and learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-setfileinformationbyhandle.
"""

from __future__ import annotations

import ctypes
import ntpath
import os
import stat
from functools import lru_cache
from types import SimpleNamespace

from mikoshi.filesystem import (
    DirectoryEntry,
    FileChangedError,
    PosixDirectory,
    UnsafePathError,
    validate_component,
)

U32 = ctypes.c_uint32
HANDLE = ctypes.c_void_p


class UnicodeString(ctypes.Structure):
    _fields_ = [("Length", ctypes.c_uint16), ("MaximumLength", ctypes.c_uint16), ("Buffer", HANDLE)]


class ObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("Length", U32),
        ("RootDirectory", HANDLE),
        ("ObjectName", ctypes.POINTER(UnicodeString)),
        ("Attributes", U32),
        ("SecurityDescriptor", HANDLE),
        ("SecurityQualityOfService", HANDLE),
    ]


class IoStatusBlock(ctypes.Structure):
    _fields_ = [("Status", HANDLE), ("Information", ctypes.c_size_t)]


class FileTime(ctypes.Structure):
    _fields_ = [("Low", U32), ("High", U32)]


class HandleInformation(ctypes.Structure):
    _fields_ = [
        ("Attributes", U32),
        ("Created", FileTime),
        ("Accessed", FileTime),
        ("Written", FileTime),
        ("Volume", U32),
        ("SizeHigh", U32),
        ("SizeLow", U32),
        ("Links", U32),
        ("IndexHigh", U32),
        ("IndexLow", U32),
    ]


class BasicInformation(ctypes.Structure):
    _fields_ = [
        ("Created", ctypes.c_int64),
        ("Accessed", ctypes.c_int64),
        ("Written", ctypes.c_int64),
        ("Changed", ctypes.c_int64),
        ("Attributes", U32),
    ]


class RenameInformation(ctypes.Structure):
    _fields_ = [
        ("ReplaceIfExists", ctypes.c_ubyte),
        ("RootDirectory", HANDLE),
        ("FileNameLength", U32),
        ("FileName", ctypes.c_uint16 * 1),
    ]


class DirectoryInformation(ctypes.Structure):
    _fields_ = [
        ("NextEntryOffset", U32),
        ("FileIndex", U32),
        ("CreationTime", ctypes.c_int64),
        ("LastAccessTime", ctypes.c_int64),
        ("LastWriteTime", ctypes.c_int64),
        ("ChangeTime", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("AllocationSize", ctypes.c_int64),
        ("FileAttributes", U32),
        ("FileNameLength", U32),
        ("EaSize", U32),
        ("ShortNameLength", ctypes.c_byte),
        ("ShortName", ctypes.c_uint16 * 12),
        ("FileId", ctypes.c_int64),
        ("FileName", ctypes.c_uint16 * 1),
    ]


@lru_cache(maxsize=1)
def _api():
    if os.name != "nt":
        raise RuntimeError("Windows handle operations require Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    native = ctypes.WinDLL("ntdll")
    native.NtCreateFile.argtypes = [
        ctypes.POINTER(HANDLE),
        U32,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        HANDLE,
        U32,
        U32,
        U32,
        U32,
        HANDLE,
        U32,
    ]
    native.NtCreateFile.restype = ctypes.c_int32
    native.NtSetInformationFile.argtypes = [
        HANDLE,
        ctypes.POINTER(IoStatusBlock),
        HANDLE,
        U32,
        ctypes.c_int,
    ]
    native.NtSetInformationFile.restype = ctypes.c_int32
    native.RtlNtStatusToDosError.argtypes = [ctypes.c_int32]
    native.RtlNtStatusToDosError.restype = U32
    signatures = {
        "CloseHandle": ([HANDLE], ctypes.c_int),
        "GetFileInformationByHandle": ([HANDLE, ctypes.POINTER(HandleInformation)], ctypes.c_int),
        "GetFileInformationByHandleEx": ([HANDLE, ctypes.c_int, HANDLE, U32], ctypes.c_int),
        "SetFileInformationByHandle": ([HANDLE, ctypes.c_int, HANDLE, U32], ctypes.c_int),
        "GetFileType": ([HANDLE], U32),
        "GetFinalPathNameByHandleW": ([HANDLE, ctypes.c_wchar_p, U32, U32], U32),
        "GetDriveTypeW": ([ctypes.c_wchar_p], U32),
    }
    for name, (arguments, result) in signatures.items():
        getattr(kernel, name).argtypes = arguments
        getattr(kernel, name).restype = result
    return kernel, native


def _raise_error(code=None):
    code = ctypes.get_last_error() if code is None else code
    if code in {32, 33, 303}:  # Sharing/lock violation or deletion pending.
        raise FileChangedError("file is busy or changing; retry the operation")
    raise ctypes.WinError(code)


def _information(handle):
    kernel, _ = _api()
    result = HandleInformation()
    if not kernel.GetFileInformationByHandle(handle, ctypes.byref(result)):
        _raise_error()
    mode = (stat.S_IFDIR | 0o777) if result.Attributes & 0x10 else (stat.S_IFREG | 0o666)
    if result.Attributes & 1:
        mode &= ~0o222
    return SimpleNamespace(
        st_mode=mode,
        st_nlink=result.Links,
        st_size=(result.SizeHigh << 32) | result.SizeLow,
        st_file_attributes=result.Attributes,
    )


def _checked_handle(
    name, root=None, *, directory=False, create=False, metadata=False, remove_readonly=False
):
    kernel, native = _api()
    length = len(name.encode("utf-16-le"))
    if length > 65532:
        raise UnsafePathError("Windows path component exceeds native name limits")
    # Windows wchar_t holds one UTF-16 code unit, not one Python character.
    buffer = ctypes.create_unicode_buffer(name, length // 2 + 1)
    encoded = UnicodeString(length, length + 2, ctypes.cast(buffer, HANDLE))
    attributes = ObjectAttributes(
        ctypes.sizeof(ObjectAttributes), root, ctypes.pointer(encoded), 0x40, None, None
    )  # OBJ_CASE_INSENSITIVE; no process-global name-resolution settings.
    handle, status = HANDLE(), IoStatusBlock()
    access = 0x00100080  # SYNCHRONIZE | FILE_READ_ATTRIBUTES
    if directory:
        access |= 0x21  # FILE_LIST_DIRECTORY | FILE_TRAVERSE
    elif not metadata:
        access |= 1  # FILE_READ_DATA
    if create:
        access |= 0x100  # FILE_WRITE_ATTRIBUTES
        if not directory:
            access |= 0x10002  # DELETE | FILE_WRITE_DATA, for staging and publication.
    if remove_readonly:
        access |= 0x10100  # DELETE | FILE_WRITE_ATTRIBUTES, without changing the ACL.
    options = 0x20  # FILE_SYNCHRONOUS_IO_NONALERT
    options |= (0x1 if directory else 0x40) if create else 0x00200000
    # Existing objects are opened without following a reparse point, then typed
    # by handle. New directories use FILE_CREATE, which never opens an existing name.
    result = native.NtCreateFile(
        ctypes.byref(handle),
        access,
        ctypes.byref(attributes),
        ctypes.byref(status),
        None,
        0x80,
        3 if directory or metadata else 1,
        2 if create else 1,
        options,
        None,
        0,
    )
    if result < 0:
        _raise_error(native.RtlNtStatusToDosError(result))
    try:
        info = _information(handle)
        if info.st_file_attributes & (0x400 | 0x40) or kernel.GetFileType(handle) != 1:
            raise UnsafePathError("reparse points and special files are not allowed in paths")
        if not metadata and stat.S_ISDIR(info.st_mode) != directory:
            raise UnsafePathError("file type changed while opening a path")
        if root is not None:
            path = ctypes.create_unicode_buffer(32768)
            size = kernel.GetFinalPathNameByHandleW(handle, path, len(path), 0)
            if not size:
                _raise_error()
            if size >= len(path):
                raise UnsafePathError("normalized Windows path exceeds native name limits")
            actual = ntpath.basename(path.value.rstrip("\\"))
            if actual.casefold() != name.casefold():
                raise UnsafePathError("Windows short-name aliases are not allowed in paths")
        return handle.value
    except BaseException:
        kernel.CloseHandle(handle)
        raise


class WindowsDirectory(PosixDirectory):
    """Pinned directory with native enumeration and handle-based publication."""

    def __init__(self, handle):
        self.handle = handle

    @classmethod
    def anchor(cls, anchor):
        # Remote shares, mapped network drives, and device namespaces do not
        # provide the local filesystem contract used by these pinned handles.
        if (
            len(anchor) != 3
            or not anchor[0].isalpha()
            or anchor[1:] != ":\\"
            or _api()[0].GetDriveTypeW(anchor) != 3
        ):
            raise UnsafePathError("Windows workspace roots must be on a local fixed drive")
        native = "\\??\\" + anchor
        return cls(_checked_handle(native, directory=True))

    def close(self):
        _api()[0].CloseHandle(self.handle)

    def info(self):
        return _information(self.handle)

    def child(self, name, *, create=False, exclusive=False):
        validate_component(name)
        if not create or not exclusive:
            try:
                return type(self)(_checked_handle(name, self.handle, directory=True))
            except FileNotFoundError:
                if not create:
                    raise
        try:
            return type(self)(_checked_handle(name, self.handle, directory=True, create=True))
        except FileExistsError:
            if exclusive:
                raise
            return type(self)(_checked_handle(name, self.handle, directory=True))

    def open_file(self, name, *, create=False, mode=0o600):
        # `mode` matches the POSIX signature; Windows files inherit the directory ACL.
        import msvcrt

        validate_component(name)
        handle = _checked_handle(name, self.handle, create=create)
        try:
            return msvcrt.open_osfhandle(
                handle, os.O_BINARY | os.O_NOINHERIT | (os.O_RDWR if create else os.O_RDONLY)
            )
        except BaseException:
            _api()[0].CloseHandle(handle)
            raise

    def entries(self):
        kernel, _ = _api()
        buffer = ctypes.create_string_buffer(65536)
        information_class = 11  # FileIdBothDirectoryRestartInfo, then continuation.
        while True:
            if not kernel.GetFileInformationByHandleEx(
                self.handle, information_class, buffer, len(buffer)
            ):
                if ctypes.get_last_error() == 18:  # ERROR_NO_MORE_FILES
                    return
                _raise_error()
            information_class = 10
            offset = 0
            while True:
                entry = DirectoryInformation.from_buffer(buffer, offset)
                start = offset + DirectoryInformation.FileName.offset
                end = start + entry.FileNameLength
                if end > len(buffer) or entry.FileNameLength % 2:
                    raise UnsafePathError("invalid Windows directory enumeration record")
                name = buffer.raw[start:end].decode("utf-16-le")
                if name not in {".", ".."}:
                    # The enumeration flags reject links before any data open.
                    mode = stat.S_IFDIR if entry.FileAttributes & 0x10 else stat.S_IFREG
                    info = SimpleNamespace(
                        st_mode=mode, st_nlink=1, st_file_attributes=entry.FileAttributes
                    )
                    if not entry.FileAttributes & 0x400:
                        handle = None
                        try:
                            validate_component(name)
                            handle = _checked_handle(name, self.handle, metadata=True)
                            info = _information(handle)
                        except FileNotFoundError:
                            info = None
                        finally:
                            if handle is not None:
                                kernel.CloseHandle(handle)
                    if info is not None:
                        yield DirectoryEntry(name, info)
                if entry.NextEntryOffset == 0:
                    break
                if (
                    entry.NextEntryOffset < DirectoryInformation.FileName.offset
                    or offset + entry.NextEntryOffset >= len(buffer)
                ):
                    raise UnsafePathError("invalid Windows directory enumeration offset")
                offset += entry.NextEntryOffset

    @staticmethod
    def file_mode(descriptor, mode):
        import msvcrt

        WindowsDirectory._mode(msvcrt.get_osfhandle(descriptor), mode)

    @staticmethod
    def _mode(handle, mode):
        kernel, _ = _api()
        info = BasicInformation()
        if not kernel.GetFileInformationByHandleEx(
            handle, 0, ctypes.byref(info), ctypes.sizeof(info)
        ):
            _raise_error()
        info.Attributes = info.Attributes & ~1 if mode & 0o222 else info.Attributes | 1
        if not kernel.SetFileInformationByHandle(
            handle, 0, ctypes.byref(info), ctypes.sizeof(info)
        ):
            _raise_error()

    def set_mode(self, mode):
        self._mode(self.handle, mode)

    def remove_readonly(self, name):
        """Remove one cleanup object through its handle, refusing links and reparses."""
        validate_component(name)
        kernel, _ = _api()
        handle = _checked_handle(name, self.handle, metadata=True, remove_readonly=True)
        try:
            self._mode(handle, 0o600)
            delete = ctypes.c_ubyte(1)
            if not kernel.SetFileInformationByHandle(handle, 4, ctypes.byref(delete), 1):
                _raise_error()
        finally:
            kernel.CloseHandle(handle)

    def _publish(self, descriptor, temporary, target, replace):
        import msvcrt

        encoded = target.encode("utf-16-le")
        # The native API requires sizeof(FILE_RENAME_INFORMATION) plus the
        # filename storage, including the structure's trailing alignment.
        buffer = ctypes.create_string_buffer(ctypes.sizeof(RenameInformation) + len(encoded))
        info = RenameInformation.from_buffer(buffer)
        info.ReplaceIfExists = replace
        info.RootDirectory = self.handle
        info.FileNameLength = len(encoded)
        ctypes.memmove(
            ctypes.addressof(buffer) + RenameInformation.FileName.offset, encoded, len(encoded)
        )
        _, native = _api()
        status = IoStatusBlock()
        result = native.NtSetInformationFile(
            msvcrt.get_osfhandle(descriptor), ctypes.byref(status), buffer, len(buffer), 10
        )  # FileRenameInformation: the target stays relative to our pinned handle.
        if result < 0:
            _raise_error(native.RtlNtStatusToDosError(result))

    def _discard_stage(self, descriptor, temporary, published):
        import msvcrt

        if not published:
            # Delete the opened staging object, never a newly substituted path.
            delete = ctypes.c_ubyte(1)
            handle = msvcrt.get_osfhandle(descriptor)
            self._mode(handle, 0o600)
            if not _api()[0].SetFileInformationByHandle(handle, 4, ctypes.byref(delete), 1):
                _raise_error()
