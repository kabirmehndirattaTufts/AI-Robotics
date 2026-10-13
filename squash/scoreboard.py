"""
Posts the score to MQTT and draws a status panel that proves it is working.

    pub = ScorePublisher("test.mosquitto.org", 1883, "ME193/Rogers/Kabir")
    pub.start()
    pub.publish_score(7)          # sends the text "7.0" (a floating-point number) to the topic
    cv2.imshow("MQTT", draw_panel(pub, record=7, now=time.time()))
    pub.stop()

The publisher also SUBSCRIBES to its own topic. When the broker sends each message back, that is
proof the whole round trip works (our program -> internet broker -> back to us), and the panel shows
the round-trip time. That is the "is MQTT functional" check for the screen recording.
"""
import random
import threading
import time
from collections import deque

import cv2
import numpy as np


class ScorePublisher:
    def __init__(self, host, port, topic, qos=0, retain=False):
        self.host, self.port, self.topic = host, port, topic
        self.qos, self.retain = qos, retain
        self.connected = False
        self.error = None                  # last connection problem, as text
        self.sent_count = 0
        self.failed_count = 0
        self.last_payload = None
        self.last_sent_t = -1e9
        self.echo_count = 0                # messages that came back from the broker
        self.echo_payload = None
        self.echo_t = -1e9
        self.echo_ms = None                # round trip time of the latest echo
        self.log = deque(maxlen=7)         # (time, "out"/"in"/"info", text)
        self._sent_at = {}                 # (topic, payload) -> time it was sent (to time the round trip)
        self.values = {}                   # name -> (payload, time sent), for the panel
        self._lock = threading.Lock()
        self._client = None

    # --- life cycle ---------------------------------------------------------------------
    def start(self):
        import paho.mqtt.client as mqtt
        cid = f"squash-{random.randint(0, 99999):05d}"
        if hasattr(mqtt, "CallbackAPIVersion"):                 # paho 2.x
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        else:                                                   # paho 1.x
            self._client = mqtt.Client(client_id=cid)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(1, 5)
        try:
            self._client.connect_async(self.host, self.port, keepalive=30)
            self._client.loop_start()
            self._note("info", f"connecting to {self.host}:{self.port} ...")
        except Exception as e:                                  # never let MQTT problems stop the game
            self.error = str(e)
            self._note("info", f"MQTT start failed: {e}")

    def stop(self):
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass

    # --- sending --------------------------------------------------------------------------
    def publish_score(self, value):
        """The main score: the record streak, as a floating-point number (7 -> "7.0"), on the main topic."""
        return self.publish("record", value, self.topic)

    def publish(self, name, value, topic=None):
        """Send one value (as a float, e.g. 7 -> "7.0"). Extra values go to <main topic>/<name>."""
        topic = topic or f"{self.topic}/{name}"
        payload = f"{float(value)}"
        now = time.time()
        if self._client is None or not self.connected:
            self.failed_count += 1
            return False
        try:
            info = self._client.publish(topic, payload, qos=self.qos, retain=self.retain)
            ok = info.rc == 0
        except Exception as e:
            self.error, ok = str(e), False
        if ok:
            with self._lock:
                self.sent_count += 1
                self.last_payload, self.last_sent_t = payload, now
                self._sent_at[(topic, payload)] = now
                self.values[name] = (payload, now)
            self._note("out", f"{topic[len(self.topic) - len(self.topic.split('/')[-1]):]}  {payload}")
        else:
            self.failed_count += 1
        return ok

    # --- callbacks (run in paho's thread) -------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc, *extra):
        ok = (not rc.is_failure) if hasattr(rc, "is_failure") else rc == 0
        if ok:
            self.connected, self.error = True, None
            client.subscribe(self.topic + "/#", qos=self.qos)  # listen to our own topics ("/#" also matches the main topic itself)
            self._note("info", "connected, listening on the topic")
        else:
            self.connected, self.error = False, f"refused ({rc})"
            self._note("info", f"connection refused: {rc}")

    def _on_disconnect(self, client, userdata, *args):
        self.connected = False
        self._note("info", "disconnected, retrying ...")

    def _on_message(self, client, userdata, msg):
        now = time.time()
        try:
            payload = msg.payload.decode()
        except Exception:
            return
        with self._lock:
            self.echo_count += 1
            self.echo_payload, self.echo_t = payload, now
            t0 = self._sent_at.get((msg.topic, payload))
            self.echo_ms = (now - t0) * 1000.0 if t0 is not None else None
        self._note("in", f"{msg.topic[len(self.topic) - len(self.topic.split(chr(47))[-1]):]}  {payload}" + (f"   ({self.echo_ms:.0f} ms)" if self.echo_ms is not None else ""))

    def _note(self, kind, text):
        with self._lock:
            self.log.append((time.time(), kind, text))


# ---------------------------------------------------------------------------------------------
# status panel
# ---------------------------------------------------------------------------------------------
GREEN, RED, AMBER = (90, 220, 90), (70, 70, 235), (0, 190, 255)
GRAY, WHITE = (150, 150, 150), (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_panel(pub, record, now, w=540, h=410):
    img = np.full((h, w, 3), 30, np.uint8)
    flash = max(0.0, 1.0 - (now - pub.last_sent_t) / 0.4)         # border flashes when a message goes out
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (int(60 + 30 * flash), int(60 + 160 * flash), int(60 + 30 * flash)), 2 + int(2 * flash))

    # header: connection status
    ok = pub.connected
    cv2.circle(img, (24, 28), 9, GREEN if ok else (AMBER if pub.error is None else RED), -1)
    cv2.putText(img, "MQTT  " + ("CONNECTED" if ok else "CONNECTING..." if pub.error is None else "NOT CONNECTED"),
                (44, 35), FONT, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, f"{pub.host}:{pub.port}", (w - 215, 35), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, f"topic  {pub.topic}", (14, 62), FONT, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    # what is being sent: one row per value, newest send flashes
    rows = (("record", "RECORD STREAK", "", (255, 220, 120)),
            ("streak", "current streak", "/streak", WHITE),
            ("points", "points", "/points", WHITE),
            ("best_points", "high score (pts)", "/best_points", WHITE))
    y = 98
    for key, label, suffix, col in rows:
        val = pub.values.get(key)
        age = (now - val[1]) if val else None
        hot = 1.0 if (age is not None and age < 0.4) else 0.0
        cv2.putText(img, label, (14, y), FONT, 0.5, GRAY if key != "record" else (255, 220, 120), 1, cv2.LINE_AA)
        cv2.putText(img, (val[0] if val else "--"), (200, y + 2), FONT, 0.8 if key != "record" else 0.95, col if not hot else GREEN, 2, cv2.LINE_AA)
        cv2.putText(img, f"{pub.topic.split('/')[-1]}{suffix}" if suffix else "(main topic)", (330, y), FONT, 0.42, GRAY, 1, cv2.LINE_AA)
        cv2.putText(img, f"{age:.1f}s ago" if age is not None else "", (450, y), FONT, 0.42, GRAY, 1, cv2.LINE_AA)
        y += 32
    cv2.putText(img, f"sent {pub.sent_count}   dropped {pub.failed_count}", (14, y + 4), FONT, 0.5, GRAY, 1, cv2.LINE_AA)

    # proof it works: the broker sends our own messages back
    y += 30
    if pub.echo_count > 0 and now - pub.echo_t < 6.0:
        ms = f"{pub.echo_ms:.0f} ms" if pub.echo_ms is not None else ""
        cv2.putText(img, f"BROKER ECHO OK  \"{pub.echo_payload}\"  {ms}   ({pub.echo_count} back)", (14, y), FONT, 0.55, GREEN, 2, cv2.LINE_AA)
    elif ok:
        cv2.putText(img, "waiting for the broker to echo it back ...", (14, y), FONT, 0.55, AMBER, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, pub.error or "no connection yet (check internet / broker)", (14, y), FONT, 0.5, RED, 1, cv2.LINE_AA)

    # message log
    cv2.line(img, (10, y + 14), (w - 10, y + 14), (70, 70, 70), 1)
    y += 36
    for t, kind, text in list(pub.log)[-6:]:
        stamp = time.strftime("%H:%M:%S", time.localtime(t))
        arrow, col = {"out": ("->", (255, 220, 120)), "in": ("<-", GREEN), "info": ("  ", GRAY)}[kind]
        cv2.putText(img, f"{stamp} {arrow} {text}"[:64], (14, y), FONT, 0.48, col, 1, cv2.LINE_AA)
        y += 19
    return img