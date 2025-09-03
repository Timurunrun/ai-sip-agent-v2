import asyncio
import json
import os
import signal
import threading
import gc
from pathlib import Path

from dotenv import load_dotenv

import pjsua2 as pj

from sip.endpoint import create_endpoint
from sip.account import Account
from sip.command_queue import CommandQueue


async def pjsua_pump(ep: pj.Endpoint, cmdq: CommandQueue, stop_event: threading.Event):
    """Pump PJSUA2 events and execute queued commands on the main thread."""
    while not stop_event.is_set():
        ep.libHandleEvents(10)          # Handle PJSIP events
        cmdq.execute_pending()          # Execute queued main-thread commands (PJSUA2 API calls)
        await asyncio.sleep(0.001)      # Give control back to loop

async def main():
    load_dotenv()

    sip_domain = os.getenv("SIP_DOMAIN")
    sip_user = os.getenv("SIP_USER")
    sip_password = os.getenv("SIP_PASSWD")
    sip_proxy = os.getenv("SIP_PROXY", "")

    if not (sip_domain and sip_user and sip_password):
        raise SystemExit("SIP creds are missing. Set SIP_DOMAIN, SIP_USER, SIP_PASSWD")

    ep = create_endpoint()
    adm = ep.audDevManager()
    adm.setNullDev()

    cmdq = CommandQueue()

    # Configure and create SIP account
    acc_cfg = pj.AccountConfig()
    acc_cfg.idUri = f"sip:{sip_user}@{sip_domain}"
    acc_cfg.regConfig.registrarUri = f"sip:{sip_domain}"
    if sip_proxy:
        # pjsua2 expects a pj.StringVector
        sv = pj.StringVector()
        sv.push_back(sip_proxy)
        acc_cfg.sipConfig.proxies = sv
    cred = pj.AuthCredInfo("digest", "*", sip_user, 0, sip_password)
    acc_cfg.sipConfig.authCreds.append(cred)

    acc = Account(cmdq)
    acc.create(acc_cfg)

    # Wait for reg in background (account signals semaphore internally)
    print("[BOOT] Waiting for SIP registration…")

    # Graceful shutdown support
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    def _handle_sig():
        print("\n[BOOT] Signal received. Shutting down…")
        stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _handle_sig())

    try:
        await pjsua_pump(ep, cmdq, stop_event)
    finally:
        # Graceful shutdown sequence
        try:
            # Hang up all calls first to trigger DISCONNECTED and onCallState cleanup
            try:
                ep.hangupAllCalls()
            except Exception:
                pass
            # Pump events longer to fully drain callbacks and queued cleanups
            for _ in range(200):  # ~2s total
                try:
                    ep.libHandleEvents(10)
                except Exception:
                    pass
                try:
                    cmdq.execute_pending()
                except Exception:
                    pass
                await asyncio.sleep(0.01)

            # Proactively delete any remaining Call objects tracked by the account
            try:
                if hasattr(acc, 'calls'):
                    for c in list(acc.calls):
                        try:
                            c.delete()
                        except Exception:
                            pass
                    acc.calls.clear()
            except Exception:
                pass

            # Delete account before destroying endpoint/lib
            try:
                acc.delete()
            except Exception:
                pass

            # Drop strong refs and force GC while lib is still alive
            try:
                del acc
            except Exception:
                pass
            gc.collect()
        finally:
            try:
                ep.libDestroy()
            finally:
                # Ensure Endpoint wrapper is also collected now
                try:
                    del ep
                except Exception:
                    pass
                gc.collect()
                print("[BOOT] Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
