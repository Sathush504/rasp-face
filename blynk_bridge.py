"""
blynk_bridge.py — Blynk IoT integration via MQTT.

Uses the official paho-mqtt library to communicate with the Blynk Cloud
MQTT broker (recommended approach for Python in 2024+).

Virtual Pin Layout (configure matching Datastreams in Blynk Console):
  V0  — Button (0/1): remote unlock trigger
  V1  — LED   (0/1): current lock state feedback
  V2  — Terminal: recent access log stream
  V3  — Label  : last authorised user name

Runs in its own daemon thread so it never blocks the GUI or recognizer.
"""

import json
import logging
import threading
import time
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt  # type: ignore
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed — Blynk integration DISABLED.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLYNK_MQTT_BROKER = "sgp1.blynk.cloud"   # SGP1 = Singapore region (match your account)
BLYNK_MQTT_PORT = 8883                    # TLS
BLYNK_MQTT_KEEPALIVE = 45

_DOWNLINK_PREFIX = "downlink/ds/"   # App → Device messages
_UPLINK_PREFIX = "ds/"              # Device → App messages


# ---------------------------------------------------------------------------
# Blynk bridge
# ---------------------------------------------------------------------------
class BlynkBridge:
    """
    Manages a persistent, auto-reconnecting MQTT connection to Blynk Cloud.

    Parameters
    ----------
    auth_token : str
        The Blynk device Auth Token (from the Blynk console).
    on_remote_unlock : callable
        Called with no args when the mobile app triggers an unlock (V1 → 1).
    vpin_unlock    : int   V-pin for remote unlock button
    vpin_status    : int   V-pin for LED status widget
    vpin_log       : int   V-pin for Terminal widget
    vpin_last_user : int   V-pin for Label widget
    """

    def __init__(
        self,
        auth_token: str,
        on_remote_unlock: Callable[[], None],
        on_command: Optional[Callable[[str], None]] = None,
        vpin_unlock: Union[int, str] = 0,
        vpin_status: Union[int, str] = 1,
        vpin_log: Union[int, str] = 2,
        vpin_last_user: Union[int, str] = 3,
    ):
        self._auth = auth_token
        self._on_remote_unlock = on_remote_unlock
        self._on_command = on_command
        self._vpin_unlock = vpin_unlock
        self._vpin_status = vpin_status
        self._vpin_log = vpin_log
        self._vpin_last_user = vpin_last_user

        self._connected = False
        self._should_run = True
        self._client: Optional["mqtt.Client"] = None  # type: ignore

        if not _MQTT_AVAILABLE:
            logger.warning("BlynkBridge is a no-op (paho-mqtt missing).")
            return

        if auth_token in ("", "YOUR_BLYNK_AUTH_TOKEN_HERE"):
            logger.warning(
                "BlynkBridge: No valid auth token — Blynk disabled. "
                "Set BLYNK_AUTH_TOKEN env var."
            )
            return

        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="BlynkMQTT"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_lock_status(self, is_unlocked: bool) -> None:
        """Push current lock state to the Blynk LED widget (V2)."""
        self._publish_vpin(self._vpin_status, 1 if is_unlocked else 0)

    def send_access_event(self, message: str) -> None:
        """Append a line to the Blynk Terminal widget (V2)."""
        self._publish_vpin(self._vpin_log, message + "\n")

    def update_last_user(self, name: str) -> None:
        """Update the last-user label widget (V4)."""
        self._publish_vpin(self._vpin_last_user, name)

    def is_connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        self._should_run = False
        if self._client:
            self._client.disconnect()

    # ------------------------------------------------------------------
    # MQTT internals
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        """Blocking loop with exponential back-off reconnect."""
        backoff = 5
        while self._should_run:
            try:
                self._connect()
                backoff = 5       # reset on successful connect
                self._client.loop_forever()
            except Exception as exc:
                logger.error("Blynk MQTT error: %s — retrying in %ds.", exc, backoff)
                self._connected = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)

    def _connect(self) -> None:
        # Client ID can be any string, using a descriptive client ID containing the token or a random suffix
        client_id = f"rasp_face_{self._auth[:6]}"
        client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv311,
            clean_session=True
        )
        # For Blynk, username is always "device", password is the Auth Token
        client.username_pw_set(username="device", password=self._auth)
        client.tls_set()            # use system CA bundle

        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message

        client.connect(BLYNK_MQTT_BROKER, BLYNK_MQTT_PORT,
                       keepalive=BLYNK_MQTT_KEEPALIVE)
        self._client = client

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("Connected to Blynk Cloud MQTT broker.")
            # Subscribe to all downlink datastream updates
            client.subscribe(f"{_DOWNLINK_PREFIX}#")
        else:
            logger.error("Blynk MQTT connection refused: rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.warning("Blynk MQTT unexpectedly disconnected: rc=%d", rc)

    def _on_message(self, client, userdata, msg) -> None:
        """Dispatch incoming messages from the Blynk app."""
        topic = msg.topic
        try:
            payload = msg.payload.decode("utf-8")
        except Exception:
            return
        
        logger.info("Received MQTT message on topic: %s | Payload: %s", topic, payload)

        # Topic format: downlink/ds/{datastream_name}
        if topic.startswith(_DOWNLINK_PREFIX):
            ds_str = topic[len(_DOWNLINK_PREFIX):]
            logger.debug("Blynk ↓ %s = %s", ds_str, payload)

            # Determine if the incoming datastream matches our unlock trigger
            is_match = False
            if isinstance(self._vpin_unlock, int):
                try:
                    is_match = (int(ds_str.lstrip("Vv")) == self._vpin_unlock)
                except ValueError:
                    pass
            else:
                # Direct string match (case-insensitive and stripping "V" if needed)
                is_match = (
                    ds_str.lower() == self._vpin_unlock.lower() or
                    ds_str.lstrip("Vv").lower() == self._vpin_unlock.lstrip("Vv").lower() or
                    (self._vpin_unlock.lower() == "unlock button" and ds_str.lower() in ("v0", "v0_button")) or
                    (self._vpin_unlock.lower() in ("v0", "v0_button") and ds_str.lower() == "unlock button")
                )

            if is_match:
                try:
                    # Payload is typically raw text "1" or "0"
                    val_int = int(float(payload))
                    if val_int == 1:
                        logger.info("Remote UNLOCK triggered via Blynk '%s'.", ds_str)
                        self._on_remote_unlock()
                except Exception as exc:
                    logger.warning("Malformed unlock payload: %s — %s", payload, exc)

            # Check if this is an incoming command from the Terminal widget
            is_log_match = False
            if isinstance(self._vpin_log, int):
                try:
                    is_log_match = (int(ds_str.lstrip("Vv")) == self._vpin_log)
                except ValueError:
                    pass
            else:
                is_log_match = (
                    ds_str.lower() == self._vpin_log.lower() or
                    ds_str.lstrip("Vv").lower() == self._vpin_log.lstrip("Vv").lower() or
                    (self._vpin_log.lower() == "access log" and ds_str.lower() in ("v2", "v2_log")) or
                    (self._vpin_log.lower() in ("v2", "v2_log") and ds_str.lower() == "access log")
                )

            if is_log_match and self._on_command:
                logger.info("Received command from Blynk terminal: %s", payload)
                self._on_command(payload.strip())

    def _publish_vpin(self, vpin, value) -> None:
        if not self._connected or self._client is None:
            return
        
        # Determine topic name based on type of vpin
        if isinstance(vpin, int):
            topic = f"{_UPLINK_PREFIX}V{vpin}"
        else:
            topic = f"{_UPLINK_PREFIX}{vpin}"

        # Blynk ds/ topics require raw string values, not JSON (which adds quotes to strings)
        payload = str(value)
        try:
            self._client.publish(topic, payload, qos=1)
        except Exception as exc:
            logger.warning("Failed to publish V%d: %s" if isinstance(vpin, int) else "Failed to publish %s: %s", vpin, exc)

    def trigger_event(self, event_code: str) -> None:
        """Trigger a Blynk push notification/event log asynchronously via HTTP API."""
        if not self._auth:
            return
        threading.Thread(
            target=self._call_log_event_api,
            args=(event_code,),
            daemon=True
        ).start()

    def _call_log_event_api(self, event_code: str) -> None:
        import urllib.request
        # Blynk API expects the token and event code
        url = (
            f"https://blynk.cloud/external/api/logEvent"
            f"?token={self._auth}&code={event_code}"
        )
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    logger.info("Successfully triggered Blynk event: %s", event_code)
                else:
                    logger.warning("Blynk event API returned status: %d", response.status)
        except Exception as exc:
            logger.warning("Failed to call Blynk event API: %s", exc)
