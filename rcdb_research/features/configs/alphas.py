from ..features import alphas101
from ..job_manager import km, t, col


alphas_config = dict(
    alphas=[
        dict(
            fn=alphas101.f1,
            dm=km(close=['close'], returns=['returns']),
        ),
        dict(
            fn=alphas101.f10,
            dm=km(close=[col('close').symlog()]),
        ),
        dict(
            fn=alphas101.f101,
            dm=km(open=['open'], high=['high'], low=['low'], close=['close']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f11,
            dm=km(close=[col('close').symlog()], vwap=[col('vwap').symlog()], volume=[col('volume').symlog()]),
        ),
        dict(
            fn=alphas101.f12,
            dm=km(close=[col('close').symlog()], volume=[col('volume').symlog()]),
        ),
        dict(
            fn=alphas101.f13,
            dm=km(close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f14,
            dm=km(open=['open'], volume=['volume'], returns=['returns']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f15,
            dm=km(high=['high'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f16,
            dm=km(high=['high'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f17,
            dm=km(close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f18,
            dm=km(open=[col('open').symlog()], close=[col('close').symlog()]),
        ),
        dict(
            fn=alphas101.f19,
            dm=km(close=['close'], returns=['returns']),
        ),
        dict(
            fn=alphas101.f2,
            dm=km(open=['open'], close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f20,
            dm=km(open=['open'], close=['close'], high=['high'], low=['low']),
        ),
        dict(
            fn=alphas101.f21,
            dm=km(close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f22,
            dm=km(close=['close'], high=['high'], volume=['volume']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f23,
            dm=km(close=['close'], high=['high']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f24,
            dm=km(close=[col('close').symlog()]),
        ),
        dict(
            fn=alphas101.f25,
            dm=km(close=['close'], high=['high'], volume=['volume'], returns=['returns'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f26,
            dm=km(high=['high'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f27,
            dm=km(volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f28,
            dm=km(
                high=[col('high').symlog()],
                low=[col('low').symlog()],
                close=[col('close').symlog()],
                volume=[col('volume').symlog()],
                returns=[col('returns').symlog()],
                vwap=[col('vwap').symlog()],
            ),
        ),
        dict(
            fn=alphas101.f29,
            dm=km(close=['close'], returns=['returns']),
        ),
        dict(
            fn=alphas101.f3,
            dm=km(open=['open'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f30,
            dm=km(close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f31,
            dm=km(low=['low'], close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f32,
            dm=km(close=[col('close').symlog()], vwap=[col('vwap').symlog()]),
        ),
        dict(
            fn=alphas101.f33,
            dm=km(open=['open'], close=['close']),
        ),
        dict(
            fn=alphas101.f34,
            dm=km(close=['close'], returns=['returns']),
        ),
        dict(
            fn=alphas101.f35,
            dm=km(high=['high'], low=['low'], close=['close'], volume=['volume'], returns=['returns']),
        ),
        dict(
            fn=alphas101.f36,
            dm=km(open=['open'], close=['close'], volume=['volume'], returns=['returns'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f37,
            dm=km(open=[col('open').symlog()], close=[col('close').symlog()]),
        ),
        dict(
            fn=alphas101.f38,
            dm=km(open=['open'], close=['close']),
        ),
        dict(
            fn=alphas101.f39,
            dm=km(close=['close'], volume=['volume'], returns=['returns']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f4,
            dm=km(low=['low']),
        ),
        dict(
            fn=alphas101.f40,
            dm=km(high=['high'], volume=['volume']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f41,
            dm=km(high=['high'], low=['low'], vwap=['vwap']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f42,
            dm=km(close=['close'], vwap=['vwap']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f43,
            dm=km(close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f44,
            dm=km(high=['high'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f45,
            dm=km(close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f46,
            dm=km(close=['close']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f47,
            dm=km(
                high=[col('high').symlog()],
                close=[col('close').symlog()],
                volume=[col('volume').symlog()],
                vwap=[col('vwap').symlog()]
            ),
        ),
        dict(
            fn=alphas101.f49,
            dm=km(close=[col('close').symlog()]),
        ),
        dict(
            fn=alphas101.f5,
            dm=km(open=['open'], close=['close'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f50,
            dm=km(volume=['volume'], vwap=['vwap']),
            tr=[t.sympower3()]
        ),
        dict(
            fn=alphas101.f51,
            dm=km(close=[col('close').symlog_symlog()]),
        ),
        dict(
            fn=alphas101.f52,
            dm=km(low=['low'], volume=['volume'], returns=['returns']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f53,
            dm=km(high=['high'], low=['low'], close=['close']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f54,
            dm=km(open=['open'], high=['high'], low=['low'], close=['close']),
        ),
        dict(
            fn=alphas101.f55,
            dm=km(high=['high'], low=['low'], close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f57,
            dm=km(close=['close'], vwap=['vwap']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f6,
            dm=km(open=['open'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f60,
            dm=km(high=['high'], low=['low'], close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f61,
            dm=km(volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f62,
            dm=km(open=['open'], high=['high'], low=['low'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f64,
            dm=km(open=['open'], high=['high'], low=['low'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f65,
            dm=km(open=['open'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f66,
            dm=km(open=['open'], high=['high'], low=['low'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f7,
            dm=km(close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f71,
            dm=km(open=['open'], volume=['volume'], low=['low'], close=['close'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f72,
            dm=km(high=['high'], volume=['volume'], low=['low'], vwap=['vwap']),
            tr=[t.symlog_symlog()]
        ),
        dict(
            fn=alphas101.f73,
            dm=km(open=['open'], low=['low'], vwap=['vwap']),
            tr=[t.symlog_symlog()]
        ),
        dict(
            fn=alphas101.f74,
            dm=km(high=['high'], close=['close'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f75,
            dm=km(low=['low'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f77,
            dm=km(high=['high'], low=['low'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f78,
            dm=km(low=['low'], volume=['volume'], vwap=['vwap']),
            tr=[t.sympower3()]
        ),
        dict(
            fn=alphas101.f8,
            dm=km(open=['open'], returns=['returns']),
        ),
        dict(
            fn=alphas101.f81,
            dm=km(volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f83,
            dm=km(high=['high'], low=['low'], close=['close'], volume=['volume'], vwap=['vwap']),
            tr=[t.symlog()]
        ),
        dict(
            fn=alphas101.f85,
            dm=km(high=['high'], low=['low'], close=['close'], volume=['volume']),
            tr=[t.sympower3()]
        ),
        dict(
            fn=alphas101.f88,
            dm=km(open=['open'], high=['high'], low=['low'], close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f9,
            dm=km(close=[col('close').symlog()]),
        ),
        dict(
            fn=alphas101.f92,
            dm=km(open=['open'], high=['high'], low=['low'], close=['close'], volume=['volume']),
        ),
        dict(
            fn=alphas101.f94,
            dm=km(volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f96,
            dm=km(close=['close'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f98,
            dm=km(open=['open'], volume=['volume'], vwap=['vwap']),
        ),
        dict(
            fn=alphas101.f99,
            dm=km(high=['high'], low=['low'], volume=['volume']),
        ),
    ]
)
