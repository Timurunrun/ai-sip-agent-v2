import asyncio
import json
import os
import signal
import threading
import gc
from pathlib import Path
import logging

from dotenv import load_dotenv

import pjsua2 as pj

from sip.endpoint import create_endpoint
from sip.account import Account
from sip.command_queue import CommandQueue
from utils.logging import setup_logging, get_logger, exception

from integrations.deepgram import DeepgramClient
from integrations.openai_structured import GPTStructuredExtractor
from integrations.amocrm import AmoCRMClient
from processing.pipeline import CallProcessingPipeline


async def pjsua_pump(ep: pj.Endpoint, cmdq: CommandQueue, stop_event: threading.Event):
    """Pump PJSUA2 events and execute queued commands on the main thread."""
    log = get_logger(__name__)
    while not stop_event.is_set():
        try:
            ep.libHandleEvents(10)          # Handle PJSIP events
        except Exception:
            exception(log, "libHandleEvents failed")
        try:
            cmdq.execute_pending()          # Execute queued main-thread commands (PJSUA2 API calls)
        except Exception:
            exception(log, "Command queue execution failed")
        await asyncio.sleep(0.001)      # Give control back to loop

async def main():
    setup_logging()
    log = get_logger("boot")
    load_dotenv()

    sip_domain = os.getenv("SIP_DOMAIN")
    sip_user = os.getenv("SIP_USER")
    sip_password = os.getenv("SIP_PASSWD")
    sip_proxy = os.getenv("SIP_PROXY", "")

    if not (sip_domain and sip_user and sip_password):
        log.error("Missing SIP creds. Set SIP_DOMAIN, SIP_USER, SIP_PASSWD")
        raise SystemExit(1)

    questions_path = Path(__file__).resolve().parent / "crm" / "questions.json"
    try:
        with questions_path.open("r", encoding="utf-8") as fp:
            questions = json.load(fp)
    except Exception as exc:
        log.error("Failed to load questions metadata", error=str(exc), path=str(questions_path))
        raise SystemExit(1) from exc

    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    amocrm_base = os.getenv("AMOCRM_BASE_URL")
    amocrm_token = os.getenv("AMOCRM_ACCESS_TOKEN")

    if not (deepgram_key and openai_key and amocrm_base and amocrm_token):
        log.error("Missing integration credentials. Set DEEPGRAM_API_KEY, OPENAI_API_KEY, AMOCRM_BASE_URL, AMOCRM_ACCESS_TOKEN")
        raise SystemExit(1)

    pipeline = CallProcessingPipeline(
        deepgram=DeepgramClient(deepgram_key, model=os.getenv("DEEPGRAM_MODEL", "nova-2")),
        extractor=GPTStructuredExtractor(
            openai_key,
            questions,
            model=os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-5-mini"),
        ),
        amocrm=AmoCRMClient(amocrm_base, amocrm_token),
        questions=questions,
        max_workers=int(os.getenv("PIPELINE_MAX_WORKERS", "12")),
    )

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

    acc = Account(cmdq, pipeline=pipeline)
    acc.create(acc_cfg)

    # Wait for reg in background (account signals semaphore internally)
    log.info("Waiting for SIP registration…")

    # Graceful shutdown support
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    def _handle_sig():
        log.info("Signal received. Shutting down…")
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
                log.info("Stopped.")
                try:
                    if pipeline:
                        pipeline.shutdown()
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.run(main())