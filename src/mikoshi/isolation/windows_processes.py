"""Win32 Job Objects and synchronous pipe polling, loaded only on Windows.

Children are assigned while CREATE_SUSPENDED is in effect, before their primary
thread can create descendants. Breakaway flags are deliberately never enabled.
See https://learn.microsoft.com/windows/win32/procthread/job-objects and
https://learn.microsoft.com/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject.
"""

from __future__ import annotations

import ctypes
import time

DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
HANDLE = ctypes.c_void_p
SIZE_T = ctypes.c_size_t


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", DWORD),
        ("MinimumWorkingSetSize", SIZE_T),
        ("MaximumWorkingSetSize", SIZE_T),
        ("ActiveProcessLimit", DWORD),
        ("Affinity", SIZE_T),
        ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", SIZE_T),
        ("JobMemoryLimit", SIZE_T),
        ("PeakProcessMemoryUsed", SIZE_T),
        ("PeakJobMemoryUsed", SIZE_T),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", DWORD),
        ("TotalProcesses", DWORD),
        ("ActiveProcesses", DWORD),
        ("TotalTerminatedProcesses", DWORD),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", DWORD),
        ("cntUsage", DWORD),
        ("th32ThreadID", DWORD),
        ("th32OwnerProcessID", DWORD),
        ("tpBasePri", LONG),
        ("tpDeltaPri", LONG),
        ("dwFlags", DWORD),
    ]


class WindowsAPI:
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], HANDLE),
            "SetInformationJobObject": ([HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD], LONG),
            "AssignProcessToJobObject": ([HANDLE, HANDLE], LONG),
            "QueryInformationJobObject": (
                [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD, ctypes.c_void_p],
                LONG,
            ),
            "TerminateJobObject": ([HANDLE, DWORD], LONG),
            "CloseHandle": ([HANDLE], LONG),
            "CreateToolhelp32Snapshot": ([DWORD, DWORD], HANDLE),
            "Thread32First": ([HANDLE, ctypes.POINTER(_ThreadEntry)], LONG),
            "Thread32Next": ([HANDLE, ctypes.POINTER(_ThreadEntry)], LONG),
            "OpenThread": ([DWORD, LONG, DWORD], HANDLE),
            "ResumeThread": ([HANDLE], DWORD),
            "PeekNamedPipe": (
                [
                    HANDLE,
                    ctypes.c_void_p,
                    DWORD,
                    ctypes.c_void_p,
                    ctypes.POINTER(DWORD),
                    ctypes.c_void_p,
                ],
                LONG,
            ),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    @staticmethod
    def error(operation):
        error = ctypes.WinError(ctypes.get_last_error())
        error.strerror = f"{operation}: {error.strerror}"
        return error

    def close(self, handle):
        if not self.kernel.CloseHandle(handle):
            raise self.error("CloseHandle")

    def resume_primary_thread(self, pid):
        # Popen closes the original primary-thread handle. The suspended process
        # has not run user code: identify that sole thread using documented APIs.
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
        if snapshot == HANDLE(-1).value:
            raise self.error("CreateToolhelp32Snapshot")
        try:
            entry = _ThreadEntry(dwSize=ctypes.sizeof(_ThreadEntry))
            threads = []
            available = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while available:
                if entry.th32OwnerProcessID == pid:
                    threads.append(entry.th32ThreadID)
                entry.dwSize = ctypes.sizeof(_ThreadEntry)
                available = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                raise self.error("Thread32Next")
            if len(threads) != 1:
                raise OSError(f"expected one suspended primary thread, observed {len(threads)}")
            thread = self.kernel.OpenThread(0x0002, False, threads[0])  # SUSPEND_RESUME
            if not thread:
                raise self.error("OpenThread")
            try:
                previous = self.kernel.ResumeThread(thread)
                if previous == 0xFFFFFFFF:
                    raise self.error("ResumeThread")
                if previous != 1:
                    raise OSError(f"unexpected primary-thread suspend count: {previous}")
            finally:
                self.close(thread)
        finally:
            self.close(snapshot)

    def available_pipe_bytes(self, descriptor):
        import msvcrt

        available = DWORD()
        handle = msvcrt.get_osfhandle(descriptor)
        if self.kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
            return available.value
        error = ctypes.get_last_error()
        if error in {109, 232, 233}:  # BROKEN_PIPE / NO_DATA / PIPE_NOT_CONNECTED
            return None
        raise self.error("PeekNamedPipe")


class WindowsJob:
    """A noninheritable, non-breakaway job; handle closure also kills its members."""

    def __init__(self):
        self.api = WindowsAPI()
        self.handle = self.api.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise self.api.error("CreateJobObjectW")
        try:
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
            if not self.api.kernel.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise self.api.error("SetInformationJobObject")
        except BaseException:
            self.close()
            raise

    def assign_and_resume(self, process):
        if not self.api.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.api.error("AssignProcessToJobObject")
        self.api.resume_primary_thread(process.pid)

    def terminate(self, process, timeout):
        # Job handles signal under selected limits, not simply on becoming empty.
        # Query active members instead of treating a signaled leader as tree exit.
        # https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information
        if not self.api.kernel.TerminateJobObject(self.handle, 1):
            raise self.api.error("TerminateJobObject")
        process.wait(timeout=timeout)
        deadline = time.monotonic() + timeout
        while True:
            accounting = _Accounting()
            if not self.api.kernel.QueryInformationJobObject(
                self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
            ):
                raise self.api.error("QueryInformationJobObject")
            active = accounting.ActiveProcesses
            if active == 0 or time.monotonic() >= deadline:
                return {"confirmed": active == 0, "scope": "job_object", "active_processes": active}
            time.sleep(0.01)

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            self.api.close(handle)
