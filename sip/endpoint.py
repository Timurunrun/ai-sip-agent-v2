import pjsua2 as pj


class _FilteredLogWriter(pj.LogWriter):
    """Custom PJSUA2 log writer to suppress noisy and unfixable known messages.

    Currently suppresses the recurring conference.c "Remove port failed"
    error (PJ_ENOTFOUND), which is benign in our teardown sequence but
    clutters the console. All other messages pass through unchanged.
    """

    def __init__(self):
        super().__init__()
        self._suppress = (
            "conference.c",
            "Remove port failed",
            "PJ_ENOTFOUND",
        )

    def write(self, entry: pj.LogEntry):
        try:
            msg = getattr(entry, "msg", "") or ""
            if all(s in msg for s in ("conference.c", "Remove port failed")) or "PJ_ENOTFOUND" in msg:
                return
            print(msg, end="")
        except Exception:
            # As a last resort, ignore logging errors entirely
            pass


def create_endpoint() -> pj.Endpoint:
    ep = pj.Endpoint()
    ep.libCreate()

    ep_cfg = pj.EpConfig()
    ep_cfg.logConfig.level = 3
    ep_cfg.logConfig.consoleLevel = 3

    # Install filtered log writer to suppress noisy benign errors
    _writer = _FilteredLogWriter()
    ep_cfg.logConfig.writer = _writer

    ep_cfg.uaConfig.maxCalls = 32
    ep_cfg.uaConfig.userAgent = "AI SIP Agent v2"
    ep_cfg.medConfig.quality = 6

    ep_cfg.medConfig.clockRate = 24000
    ep_cfg.medConfig.sndClockRate = 24000
    ep_cfg.medConfig.audioFramePtime = 20
    ep_cfg.medConfig.ecOptions = 0
    ep_cfg.medConfig.ecTailLen = 0
    ep_cfg.medConfig.jbInit = 20
    ep_cfg.medConfig.jbMinPre = 10
    ep_cfg.medConfig.jbMaxPre = 100

    ep.libInit(ep_cfg)
    setattr(ep, "_log_writer", _writer)

    transport_cfg = pj.TransportConfig()
    transport_cfg.port = 5060
    ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
    ep.libStart()

    return ep
