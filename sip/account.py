import os
import re
import time
from pathlib import Path
import threading

import pjsua2 as pj

from .call import Call
from utils.logging import get_logger, bind, exception


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
        self.log = get_logger("sip.account")

    def onRegState(self, prm):
        self.log.info("Registration", reason=prm.reason)
        if prm.reason == "Ok":
            self.sem_reg.release()

    def onIncomingCall(self, prm):
        call = Call(self, prm.callId)
        # Hold a strong reference so Python GC doesn't destroy it prematurely
        self.calls.append(call)
        ci = call.getInfo()
        # Extract phone number
        m = re.search(r"sip:([^@>]+)@", ci.remoteUri)
        phone = m.group(1) if m else None
        call_id = getattr(ci, "callIdString", None) or str(id(call))
        clog = bind(self.log, call_id=call_id, remote=ci.remoteUri, phone=str(phone or "?"))
        clog.info("Incoming call")

        if "sipvicious" in ci.remoteUri.lower():
            clog.warning("Ignoring SIPVicious scan")
            return

        if not phone:
            clog.warning("Cannot parse phone — ignoring")
            return
        digits_only = re.sub(r'\D', '', phone)
        if len(digits_only) != 11:
            clog.warning("Non-RU phone — ignoring", digits=digits_only)
            return

        # Create our Call object and answer from main thread
        ts = int(time.time())
        rec_path = TMP_DIR / f"call_{ts}.wav"
        call.prepare_recording(str(rec_path))
        clog.info("Prepared recording", path=str(rec_path))

        def _answer():
            prm = pj.CallOpParam()
            prm.statusCode = 200
            try:
                call.answer(prm)
                clog.info("Answered")
            except Exception as e:
                exception(clog, "Answer failed")

        self.cmdq.put(_answer)