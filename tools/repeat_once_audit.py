"""Observe existing operations; never call restore/encode for measurement."""
from collections import deque
import json

import torch
from kvq.quant import Quantized


class InvalidExecution(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidExecution(message)


def compare_bits(x, y):
    result = dict(shape=list(x.shape), reference_shape=list(y.shape),
                  dtype=str(x.dtype), reference_dtype=str(y.dtype))
    if x.shape != y.shape or x.dtype != y.dtype:
        return dict(result, equal=False, reason='shape_or_dtype')
    x = x.detach().cpu().contiguous()
    y = y.detach().cpu().contiguous()
    require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), 'nonfinite latent')
    different = x.view(torch.uint8) != y.view(torch.uint8)
    per_element = different.reshape(*x.shape, x.element_size()).any(-1)
    indices = per_element.nonzero()
    result.update(equal=not bool(per_element.any()), different_bytes=int(different.sum()),
                  different_elements=int(per_element.sum()),
                  first_index=indices[0].tolist() if len(indices) else None,
                  max_abs_difference=float((x.float()-y.float()).abs().max()))
    if x.ndim >= 2 and x.shape[1] % 3 == 0:
        result['different_chunks'] = [i for i in range(x.shape[1]//3)
                                     if bool(per_element[:, 3*i:3*i+3].any())]
    return result


class OperationAudit:
    """One audit per process; references are cleared at every attend return.

    Restore outputs are observed by object identity, not recomputed or hashed.
    The target Quantizer.encode wrapper counts real scalar-kind invocations.
    """
    def __init__(self, stores, event_path=None, expected_chunks=42, calls_per_chunk=5):
        self.stores = stores
        self.expected_chunks = expected_chunks
        self.calls_per_chunk = calls_per_chunk
        self.repeat_expected = [s.repeat for s in stores]
        self.current = None
        self.restore_original = None
        self.file = open(event_path, 'w', buffering=1024*1024) if event_path else None
        self.records = [dict(layer=i, attention_calls=0, pair_calls=0,
                             fresh_k=0, fresh_v=0, repeat_k=0, repeat_v=0,
                             returned_k=0, returned_v=0, restore_k=0, restore_v=0)
                        for i in range(len(stores))]
        self.originals = []

    def emit(self, row):
        if self.file:
            self.file.write(json.dumps(row, separators=(',', ':'))+'\n')

    def __enter__(self):
        self.restore_original = Quantized.restore
        owner = self
        def restore(state):
            value = owner.restore_original(state)
            ctx = owner.current
            if ctx is not None:
                ctx['recent'].append((state, value))
                ctx['restore_total'] += 1
            return value
        Quantized.restore = restore
        for layer, store in enumerate(self.stores):
            self.install(layer, store)
        return self

    def install(self, layer, store):
        owner = self
        old_attend, old_pair = store.attend, store.quantizer.encode_pair
        target = store.quantizer.target
        old_encode = target.encode
        self.originals.append((store, old_attend, old_pair, target, old_encode))
        require(not store.quantizer.observe, 'statistics mode would add restore calls')
        require((target.method, target.key_bits, target.value_bits, target.group) == ('rtn',4,4,16),
                'quantizer configuration changed')

        def scalar_encode(x, *, value=False):
            ctx = owner.current
            require(ctx is not None and ctx['layer'] == layer and ctx['active_pair'] is not None,
                    'untracked encode invocation')
            active = ctx['active_pair']
            kind = 'v' if value else 'k'
            require(x is active['inputs'][int(value)], 'encode_pair changed input object')
            require(active['observed'] == ([] if not value else ['k']), 'encode order/count changed')
            active['observed'].append(kind)
            owner.records[layer][active['operation']+'_'+kind] += 1
            owner.emit(dict(event='encode_enter', layer=layer, chunk=ctx['chunk'], call=ctx['call'],
                            segment=active['start'], operation=active['operation'], kind=kind,
                            shape=list(x.shape), dtype=str(x.dtype), group=target.group,
                            input_provenance=active['provenance']))
            result = old_encode(x, value=value)
            owner.records[layer]['returned_'+kind] += 1
            owner.emit(dict(event='encode_return', layer=layer, chunk=ctx['chunk'], call=ctx['call'],
                            segment=active['start'], operation=active['operation'], kind=kind))
            return result

        def pair(k, v):
            ctx = owner.current
            require(ctx is not None and ctx['layer'] == layer and ctx['queue'], 'unexpected pair encode')
            operation, start, previous = ctx['queue'].popleft()
            if operation == 'repeat':
                recent = list(ctx['recent'])
                require(len(recent)==2 and all(a is b for a,b in zip(previous, (r[0] for r in recent))),
                        'repeat input did not come from expected segment states')
                require(k is recent[0][1] and v is recent[1][1], 'repeat input is not actual restore return')
                provenance = 'actual_restore_return_object'
            else:
                require(k is previous[0] and v is previous[1], 'fresh input is not previous pending')
                provenance = 'previous_pending_object'
            require(k.dtype == v.dtype == torch.bfloat16, 'encode dtype changed')
            require(k.shape[1] == v.shape[1] == ctx['count'], 'encode chunk length changed')
            ctx['active_pair'] = dict(operation=operation, start=start, inputs=(k,v), observed=[],
                                      provenance=provenance)
            result = old_pair(k, v)
            require(ctx['active_pair']['observed']==['k','v'], 'missing actual K/V encode')
            owner.records[layer]['pair_calls'] += 1
            ctx['active_pair'] = None
            ctx['recent'].clear()
            return result

        def attend(query, key, value, start, attention, *, local=True, record=None):
            require(owner.current is None, 'nested audited attention')
            require(store.repeat == owner.repeat_expected[layer], 'storage arm changed during execution')
            row = owner.records[layer]
            invocation = row['attention_calls']
            chunk, call = divmod(invocation, owner.calls_per_chunk)
            count = key.shape[1]
            require(count == 4680 and start == chunk*count and local and record is None,
                    'attention schedule/configuration changed')
            require(store.capacity == 7*count, 'cache capacity changed')
            transitioned = store.pending_start is not None and start != store.pending_start
            survivors = list(store.segments)
            if transitioned:
                while len(survivors) >= 6:
                    survivors.pop(0)
            queue = deque()
            if store.repeat:
                queue.extend(('repeat', s['start'], s['state']) for s in survivors)
            if transitioned:
                queue.append(('fresh', store.pending_start, store.pending))
            # Do not retain obsolete quantized states beyond their actual re-encode.
            del survivors
            ctx = dict(layer=layer, chunk=chunk, call=call, count=count, queue=queue,
                       recent=deque(maxlen=2), active_pair=None, restore_total=0)
            owner.current = ctx
            delivered = []
            def observed_attention(q, k, v):
                require(not ctx['queue'] and ctx['active_pair'] is None, 'attention before expected encodes')
                expected_starts = [c*count for c in range(max(0,chunk-6),chunk)]
                require([s['start'] for s in store.segments] == expected_starts, 'eviction/order changed')
                expected_counts = [1+(5*(chunk-s//count-1)+call if store.repeat else 0)
                                   for s in expected_starts]
                require([s['encodes'] for s in store.segments] == expected_counts, 'segment encode history changed')
                require(store.pending_start==start and q is query, 'pending/query changed')
                require(q.dtype == k.dtype == v.dtype == torch.bfloat16, 'attention dtype changed')
                require(k.shape == v.shape and k.shape[1]==(min(chunk,6)+1)*count,
                        'attention sequence changed')
                delivered.append(True)
                owner.emit(dict(event='attention', layer=layer, chunk=chunk, call=call,
                                history_starts=expected_starts, history_encodes=expected_counts,
                                pending_start=start, tokens=k.shape[1], dtype=str(k.dtype),
                                repeat=store.repeat))
                ctx['recent'].clear()
                return attention(q,k,v)
            try:
                result = old_attend(query,key,value,start,observed_attention,local=local,record=record)
                require(len(delivered)==1, 'attention call count changed')
                require(ctx['restore_total']%2==0, 'unpaired restore')
                row['restore_k'] += ctx['restore_total']//2
                row['restore_v'] += ctx['restore_total']//2
                row['attention_calls'] += 1
                return result
            finally:
                owner.current = None
                ctx['recent'].clear()
        target.encode = scalar_encode
        store.quantizer.encode_pair = pair
        store.attend = attend

    def validate(self):
        chunks, steps = self.expected_chunks, self.calls_per_chunk
        fresh = chunks-1
        reads = steps*sum(min(c,6) for c in range(chunks))
        for store, row in zip(self.stores, self.records):
            total = reads if store.repeat else fresh
            repeat = total-fresh
            require(row['attention_calls']==chunks*steps, 'incomplete attention calls')
            for kind in ('k','v'):
                require(row['fresh_'+kind]==fresh and row['repeat_'+kind]==repeat,
                        f'layer {row["layer"]} {kind} encode counts differ')
                require(row['returned_'+kind]==total, 'encode did not return')
                require(row['restore_'+kind]==reads+repeat, 'extra/missing actual restore calls')
            require(row['pair_calls']==total==store.quantizer.calls, 'pair counter mismatch')
        return dict(valid=True, layers=self.records,
                    scope='actual Quantizer.encode entry/return; restore-return object provenance; no added codec calls')

    def __exit__(self, *_):
        Quantized.restore = self.restore_original
        for store, attend, pair, target, encode in self.originals:
            store.attend, store.quantizer.encode_pair, target.encode = attend, pair, encode
        if self.file:
            self.file.close()
