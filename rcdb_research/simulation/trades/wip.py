#
# def __str__(self):
#     order_params = str(self.order.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
#     order_str = f"{self.order.__class__.__name__}({order_params})"
#     d = self.to_dict()
#     d['order'] = order_str
#     params = str(d).strip("{}").replace(': ', '=').replace("'", '')
#     return f"{self.__class__.__name__}({params})"
#
# def to_dict(self, prefix=""):
#     d = dict(self._asdict())
#
#     d = {k: v.to_dict() for (k, v) in d.items() if v is not None}
#
#     return d

#     def to_df(self):
#         d = dict(self._asdict())

#         nd = OrderedDict()
#         nd.update(d['pre_trade_state'].to_dict(prefix="pre_"))

#         trade_signal = d.get('trade_signal', None)
#         if trade_signal is not None:
#             nd.update(trade_signal.to_dict())

#         order = d.get('order', None)
#         if order is not None:
#             nd.update({(f"order_{k}" if k == 'type' else k): v for (k, v) in order.to_dict().items()})

#         execution_result = d.get('execution_result', None)
#         if execution_result is not None:
#             nd.update({k: v for (k, v) in execution_result.to_dict().items() if k != 'order'})

#         nd.update(d['post_trade_state'].to_dict(prefix="post_"))

#         return pd.DataFrame([nd])
