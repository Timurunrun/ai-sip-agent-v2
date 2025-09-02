import pjsua2 as pj


def create_endpoint() -> pj.Endpoint:
    ep = pj.Endpoint()
    ep.libCreate()

    ep_cfg = pj.EpConfig()
    ep_cfg.logConfig.level = 3
    ep_cfg.logConfig.consoleLevel = 3
    ep_cfg.uaConfig.maxCalls = 32
    ep_cfg.uaConfig.userAgent = "AI SIP Agent v2"

    # Media tuning for low latency
    ep_cfg.medConfig.quality = 6
    # Use 24 kHz internal clock so recorded media is 24k for realtime API
    try:
        ep_cfg.medConfig.clockRate = 24000
    except Exception:
        pass
    ep_cfg.medConfig.sndClockRate = 24000
    ep_cfg.medConfig.audioFramePtime = 20
    ep_cfg.medConfig.ecOptions = 0
    ep_cfg.medConfig.ecTailLen = 0
    ep_cfg.medConfig.jbInit = 20
    ep_cfg.medConfig.jbMinPre = 10
    ep_cfg.medConfig.jbMaxPre = 100

    ep.libInit(ep_cfg)

    transport_cfg = pj.TransportConfig()
    transport_cfg.port = 5060
    ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
    ep.libStart()

    return ep
