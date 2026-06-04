"""
hardware.py — Hardware Abstraction Layer (HAL) for the door lock solenoid.

On a Raspberry Pi the real RPi.GPIO library is used.
On any other platform (e.g. a Fedora dev laptop) a software stub is activated
automatically so the rest of the code can run without physical hardware.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detect hardware capability
# ---------------------------------------------------------------------------
try:
    import RPi.GPIO as GPIO  # type: ignore
    _GPIO_AVAILABLE = True
    logger.info("RPi.GPIO detected — running in hardware mode.")
except ImportError:
    _GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available — running in SOFTWARE SIMULATION mode.")


# ---------------------------------------------------------------------------
# Stub GPIO for non-Pi environments
# ---------------------------------------------------------------------------
class _StubGPIO:
    """Mimics the subset of RPi.GPIO used by this project."""
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    HIGH = True
    LOW = False

    def setmode(self, mode):
        logger.debug("[STUB GPIO] setmode(%s)", mode)

    def setup(self, pin, direction, initial=None):
        kwargs = f", initial={initial}" if initial is not None else ""
        logger.debug("[STUB GPIO] setup(pin=%s, dir=%s%s)", pin, direction, kwargs)

    def output(self, pin, value):
        state = "HIGH" if value else "LOW"
        logger.info("[STUB GPIO] pin %s → %s", pin, state)

    def input(self, pin):
        logger.debug("[STUB GPIO] input(pin=%s) → 0", pin)
        return 0

    def cleanup(self):
        logger.debug("[STUB GPIO] cleanup()")


if not _GPIO_AVAILABLE:
    GPIO = _StubGPIO()  # type: ignore


# ---------------------------------------------------------------------------
# DoorLock controller
# ---------------------------------------------------------------------------
class DoorLock:
    """
    Controls the solenoid door lock via a GPIO pin.

    Parameters
    ----------
    pin : int
        BCM GPIO pin number wired to the relay/solenoid gate.
    active_high : bool
        If True, HIGH = unlocked. If False, LOW = unlocked (active-low relay).
    unlock_duration : float
        Seconds the lock stays open before auto-relocking.
    """

    def __init__(self, pin: int, active_high: bool = True,
                 unlock_duration: float = 3.0, remote_ip: str = None):
        self.pin = pin
        self.active_high = active_high
        self.unlock_duration = unlock_duration
        self._lock_state = False          # False = locked
        self._relock_timer: threading.Timer | None = None
        self._state_lock = threading.Lock()
        self._on_state_change_callback = None

        # Load from config if not explicitly passed
        if remote_ip is None:
            try:
                from config import REMOTE_GPIO_IP
                self.remote_ip = REMOTE_GPIO_IP
            except ImportError:
                self.remote_ip = None
        else:
            self.remote_ip = remote_ip

        self._pi = None
        if self.remote_ip:
            logger.info("DoorLock initializing in REMOTE GPIO mode on Pi at %s", self.remote_ip)
            try:
                import pigpio
                self._pi = pigpio.pi(self.remote_ip)
                if not self._pi.connected:
                    logger.warning("Could not connect to remote pigpiod daemon at %s. Falling back to local/simulation mode.", self.remote_ip)
                    self._pi = None
                else:
                    self._pi.set_mode(self.pin, pigpio.OUTPUT)
            except Exception as exc:
                logger.warning("Failed to initialize remote pigpio to %s: %s. Falling back to local/simulation mode.", self.remote_ip, exc)
                self._pi = None

        if not self._pi:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=self._locked_signal())
            logger.info("DoorLock initialised on local GPIO pin %d.", self.pin)
        else:
            self._set_gpio(unlocked=False)
            logger.info("DoorLock initialised on remote GPIO pin %d.", self.pin)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def unlock(self, triggered_by: str = "system") -> None:
        """Unlock the door for `unlock_duration` seconds, then auto-relock."""
        with self._state_lock:
            self._cancel_relock_timer()
            self._set_gpio(unlocked=True)
            self._lock_state = True
            logger.info("DOOR UNLOCKED  ← triggered by: %s", triggered_by)
            self._relock_timer = threading.Timer(
                self.unlock_duration, self._auto_relock
            )
            self._relock_timer.daemon = True
            self._relock_timer.start()
            self._fire_callback()

    def lock(self) -> None:
        """Immediately re-lock the door."""
        with self._state_lock:
            self._cancel_relock_timer()
            self._set_gpio(unlocked=False)
            self._lock_state = False
            logger.info("DOOR LOCKED")
            self._fire_callback()

    @property
    def is_unlocked(self) -> bool:
        return self._lock_state

    def set_state_change_callback(self, callback) -> None:
        """Register a callback(is_unlocked: bool) fired on every state change."""
        self._on_state_change_callback = callback

    def cleanup(self) -> None:
        """Release GPIO resources. Call on application exit."""
        self._cancel_relock_timer()
        if self._pi:
            try:
                self._pi.stop()
                logger.info("Remote GPIO connection closed.")
            except Exception:
                pass
        else:
            GPIO.cleanup()
            logger.info("GPIO cleaned up.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _locked_signal(self) -> bool:
        return not self.active_high   # LOW if active_high, HIGH if active_low

    def _unlocked_signal(self) -> bool:
        return self.active_high

    def _set_gpio(self, unlocked: bool) -> None:
        signal = self._unlocked_signal() if unlocked else self._locked_signal()
        if self._pi:
            val = 1 if signal else 0
            try:
                self._pi.write(self.pin, val)
                logger.debug("[REMOTE GPIO] pin %d → %d", self.pin, val)
            except Exception as exc:
                logger.warning("Remote GPIO write failed: %s", exc)
        else:
            GPIO.output(self.pin, GPIO.HIGH if signal else GPIO.LOW)

    def _auto_relock(self) -> None:
        with self._state_lock:
            self._set_gpio(unlocked=False)
            self._lock_state = False
            logger.info("DOOR AUTO-RELOCKED after %ss.", self.unlock_duration)
            self._fire_callback()

    def _cancel_relock_timer(self) -> None:
        if self._relock_timer and self._relock_timer.is_alive():
            self._relock_timer.cancel()
            self._relock_timer = None

    def _fire_callback(self) -> None:
        if self._on_state_change_callback:
            try:
                self._on_state_change_callback(self._lock_state)
            except Exception:
                logger.exception("Error in DoorLock state-change callback.")
