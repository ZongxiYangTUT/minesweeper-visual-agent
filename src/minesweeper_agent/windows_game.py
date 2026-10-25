"""Window management only; gameplay state always comes from screen pixels."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys
import time

from .capture import screenshot
from .controller import execute
from .geometry import cell_center
from .model import Move, Rect, Settings


class WindowsGame:
    def __init__(self, executable: Path):
        if sys.platform != "win32":
            raise OSError("Live play requires Windows")
        self.executable = executable.resolve(strict=True)
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user.GetForegroundWindow.restype = wintypes.HWND
        self.user.GetLastActivePopup.argtypes = [wintypes.HWND]
        self.user.GetLastActivePopup.restype = wintypes.HWND
        self.user.WindowFromPoint.argtypes = [wintypes.POINT]
        self.user.WindowFromPoint.restype = wintypes.HWND
        self.user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.user.GetAncestor.restype = wintypes.HWND
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                         wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.user.WaitForInputIdle.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        for name in ("IsWindowVisible", "IsWindow", "SetForegroundWindow"):
            getattr(self.user, name).argtypes = [wintypes.HWND]
        self.user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user.GetAsyncKeyState.restype = ctypes.c_short
        self.user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int, wintypes.UINT]
        self.user.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.user.SendMessageW.restype = wintypes.LPARAM
        self.hwnd = None
        self.expected_rect: Rect | None = None

    def _find(self):
        matches = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        @callback_type
        def visit(hwnd, _):
            if not self.user.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process = self.kernel.OpenProcess(0x1000, False, pid.value)  # query executable path only
            if process:
                try:
                    size = wintypes.DWORD(32768)
                    buf = ctypes.create_unicode_buffer(size.value)
                    if self.kernel.QueryFullProcessImageNameW(process, 0, buf, ctypes.byref(size)):
                        if Path(buf.value) == self.executable:
                            # Owned dialogs are not the main game window.
                            if self.user.GetAncestor(hwnd, 3) == hwnd:  # GA_ROOTOWNER
                                matches.append(hwnd)
                finally:
                    self.kernel.CloseHandle(process)
            return True

        self.user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        self.user.EnumWindows(visit, 0)
        if len(matches) > 1:
            raise ValueError("Multiple game windows are open; close duplicates first")
        return matches[0] if matches else None

    def open(self):
        self.hwnd = self._find()
        if self.hwnd is None:
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 1  # an interactive game, even when this CLI has a hidden console
            process = subprocess.Popen([str(self.executable)], cwd=str(self.executable.parent), startupinfo=startup)
            handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, process.pid)
            if handle:
                try:
                    self.user.WaitForInputIdle(handle, 5000)
                finally:
                    self.kernel.CloseHandle(handle)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and self.hwnd is None:
                time.sleep(.2)
                self.hwnd = self._find()
        if self.hwnd is None:
            raise OSError("The game did not expose a window")
        # Startup activation is asynchronous for this legacy executable. A window
        # can be enumerated before its final initial ShowWindow has taken effect.
        for attempt in range(2):
            self.user.ShowWindow(self.hwnd, 9)  # restore
            popup = self.user.GetLastActivePopup(self.hwnd)
            self.user.SetForegroundWindow(popup or self.hwnd)
            time.sleep(.3)
            if self.user.GetAncestor(self.user.GetForegroundWindow(), 3) != self.hwnd:
                self._focus_on_startup()
            for _ in range(2):
                if not self.dismiss_record_dialog():
                    break
            try:
                self.assert_ready(check_geometry=False)
                return
            except OSError:
                if attempt:
                    raise
                time.sleep(.6)

    def _focus_on_startup(self):
        """Windows can reject SetForegroundWindow for a background-launched CLI.

        Briefly raise only the selected game, then click a verified caption point.
        Restore its topmost status immediately. Never re-steal focus during play.
        """
        import pyautogui
        was_topmost = bool(self.user.GetWindowLongW(self.hwnd, -20) & 8)
        try:
            self.user.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 0x13)
            time.sleep(.15)
            rect = wintypes.RECT()
            if not self.user.GetWindowRect(self.hwnd, ctypes.byref(rect)):
                raise OSError("Game closed during activation")
            x, y = rect.left + 36, rect.top + 16
            hit = self.user.WindowFromPoint(wintypes.POINT(x, y))
            caption = self.user.SendMessageW(self.hwnd, 0x84, 0, ((y & 0xffff) << 16) | (x & 0xffff))
            if self.user.GetAncestor(hit, 2) != self.hwnd or caption != 2:
                raise OSError("Cannot safely focus the game caption")
            if self.user.GetAsyncKeyState(0x1B) & 0x8000:
                raise KeyboardInterrupt
            pyautogui.click(x, y)
            time.sleep(.15)
        finally:
            if not was_topmost:
                self.user.SetWindowPos(self.hwnd, -2, 0, 0, 0, 0, 0x13)

    def client_rect(self) -> Rect:
        rect, point = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.user.GetClientRect(self.hwnd, ctypes.byref(rect)):
            raise OSError("Game window closed")
        if not self.user.ClientToScreen(self.hwnd, ctypes.byref(point)):
            raise OSError("Cannot locate game window")
        return Rect(point.x, point.y, rect.right, rect.bottom)

    def assert_ready(self, check_geometry=True):
        if self.user.GetAsyncKeyState(0x1B) & 0x8000:
            raise KeyboardInterrupt
        if self.user.GetForegroundWindow() != self.hwnd:
            raise OSError("Game lost focus or a dialog opened; stopped without further input")
        rect = self.client_rect()
        if check_geometry and self.expected_rect is not None and rect != self.expected_rect:
            raise OSError("Game window moved or resized; stopped before clicking")
        # Check several points so an overlapping window is not interpreted as the board.
        for x, y in ((rect.left + 2, rect.top + 2),
                     (rect.left + rect.width - 3, rect.top + rect.height - 3),
                     (rect.left + rect.width // 2, rect.top + rect.height // 2)):
            hit = self.user.WindowFromPoint(wintypes.POINT(x, y))
            if self.user.GetAncestor(hit, 2) != self.hwnd:
                raise OSError("Game is covered by another window")

    def dismiss_record_dialog(self) -> bool:
        """Accept only XP's observed local high-score dialog, preserving its default name.

        Reads visible control captions, never game memory or hidden board contents.
        All other dialogs and focus changes remain a stop condition.
        """
        import pyautogui
        foreground = self.user.GetForegroundWindow()
        if not foreground or self.user.GetAncestor(foreground, 3) != self.hwnd:
            return False
        captions = []
        dialogs = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def is_dialog(hwnd):
            name = ctypes.create_unicode_buffer(256)
            self.user.GetClassNameW(hwnd, name, len(name))
            return name.value == "#32770" and self.user.IsWindowVisible(hwnd)

        @callback_type
        def find_dialog(hwnd, _):
            if is_dialog(hwnd) and self.user.GetAncestor(hwnd, 3) == self.hwnd:
                dialogs.append(hwnd)
            return True

        self.user.EnumChildWindows.argtypes = [wintypes.HWND, callback_type, wintypes.LPARAM]
        self.user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        self.user.EnumWindows(find_dialog, 0)
        if is_dialog(foreground) and foreground not in dialogs:
            dialogs.append(foreground)
        if len(dialogs) != 1:
            return False

        @callback_type
        def visit(hwnd, _):
            buf = ctypes.create_unicode_buffer(1024)
            self.user.GetWindowTextW(hwnd, buf, len(buf))
            captions.append(buf.value)
            return True

        self.user.EnumChildWindows(dialogs[0], visit, 0)
        text = " ".join(captions)
        title = ctypes.create_unicode_buffer(256)
        self.user.GetWindowTextW(dialogs[0], title, len(title))
        record = "已破" in text and "请留尊姓大名" in text
        scores = title.value == "扫雷英雄榜" and "重新计分" in text and "确定" in text
        if not (record or scores):
            return False
        if self.user.GetAsyncKeyState(0x1B) & 0x8000:
            raise KeyboardInterrupt
        if self.user.GetForegroundWindow() != foreground:
            raise OSError("Focus changed before dismissing the record dialog")
        self.user.SetForegroundWindow(dialogs[0])
        time.sleep(.1)
        if self.user.GetForegroundWindow() != dialogs[0]:
            raise OSError("Could not focus the recognised record dialog")
        pyautogui.press("enter")
        time.sleep(.3)
        return True

    def new_game(self, difficulty: str):
        import pyautogui
        for _ in range(2):
            if not self.dismiss_record_dialog():
                break
        self.assert_ready(check_geometry=False)
        if difficulty != "current":
            pyautogui.hotkey("alt", "g")
            time.sleep(.15)
            pyautogui.press({"beginner": "b", "intermediate": "i", "expert": "e"}[difficulty])
            time.sleep(.2)
        self.assert_ready(check_geometry=False)
        pyautogui.press("f2")
        time.sleep(.3)
        self.expected_rect = self.client_rect()
        self.assert_ready()

    def capture(self):
        for _ in range(2):
            if not self.dismiss_record_dialog():
                break
        self.assert_ready()
        return screenshot(self.client_rect())

    def click(self, move: Move, settings: Settings):
        self.assert_ready()
        x, y = cell_center(settings.rect, settings.rows, settings.cols, move.row, move.col)
        hit = self.user.WindowFromPoint(wintypes.POINT(x, y))
        if self.user.GetAncestor(hit, 2) != self.hwnd:
            raise OSError("Click target is covered by another window")
        return execute(move, settings)
