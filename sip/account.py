import os
import re
import time
from pathlib import Path
import threading

import pjsua2 as pj

from .call import Call


TMP_DIR = Path("/tmp/pjsua_recordings_v2")
TMP_DIR.mkdir(parents=True, exist_ok=True)


class Account(pj.Account):
    """SIP account that accepts incoming calls and creates Call handlers."""

    def __init__(self, cmdq):
        super().__init__()
        self.cmdq = cmdq
        self.sem_reg = threading.Semaphore(0)
        # Track active Call objects to delete them before shutdown
        self.calls: list[Call] = []

    def onRegState(self, prm):
        print(f"[SIP] Registration: {prm.reason}")
        if prm.reason == "Ok":
            self.sem_reg.release()

    def onIncomingCall(self, prm):
        call = Call(self, prm.callId)
        # Hold a strong reference so Python GC doesn't destroy it prematurely
        self.calls.append(call)
        ci = call.getInfo()
        print(f"[SIP] Incoming call from: {ci.remoteUri}")

        if "sipvicious" in ci.remoteUri.lower():
            print("[SIP] Ignoring SIPVicious")
            return

        # Extract phone number
        m = re.search(r"sip:([^@>]+)@", ci.remoteUri)
        phone = m.group(1) if m else None
        if not phone:
            print("[SIP] Cannot parse phone — ignoring")
            return
        digits_only = re.sub(r'\D', '', phone)
        if len(digits_only) != 11:
            print(f"[PJSUA] Обнаружен не российский номер ({digits_only}) — игнорируем")
            return

        # Create our Call object and answer from main thread
        ts = int(time.time())
        rec_path = TMP_DIR / f"call_{ts}.wav"
        call.prepare_recording(str(rec_path))

        def _answer():
            prm = pj.CallOpParam()
            prm.statusCode = 200
            try:
                call.answer(prm)
                print("[SIP] Answered")
            except Exception as e:
                print(f"[SIP] Answer error: {e}")

        self.cmdq.put(_answer)
