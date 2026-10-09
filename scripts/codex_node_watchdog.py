"""Keep the mihomo `主代理` pinned to a node that can actually serve OpenAI Codex.

Runs as a systemd oneshot (see `codex-node-watchdog.timer`, every 10 minutes).
Each run:

1. probe the currently pinned node with an authenticated
   ``GET https://chatgpt.com/backend-api/codex/models`` (200 = healthy);
2. if the tunnel is fine, exit without touching anything;
3. otherwise scan every member of the group for latency, then try the fastest
   candidates in order and pin the first one whose Codex endpoint answers.

Nodes that merely *connect* are not enough: e.g. the Hong Kong node has the
lowest ping but gets a Cloudflare 403 on chatgpt.com, so it is rejected here.

Probe verdicts: ``ok`` (200), ``account`` (401 invalid_api_key -- the tunnel and
the edge work, the problem is on the account side, so switching nodes will not
help), ``blocked`` (any other HTTP status), ``dead`` (timeout / connection error).

Exit codes: 0 = healthy (possibly after re-pinning), 1 = no usable node,
2 = fatal (controller unreachable / credentials unusable).
"""

import argparse
import concurrent.futures
import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_CONTROLLER = 'http://127.0.0.1:9090'
DEFAULT_PROXY = 'http://127.0.0.1:7892'
DEFAULT_GROUP = '主代理'
DEFAULT_CODEX_HOME = os.path.expanduser('~/.codex')
PROBE_HOST = 'https://chatgpt.com'
PROBE_PATH = '/backend-api/codex/models?client_version=0.155.0'
PROBE_URL = PROBE_HOST + PROBE_PATH
DELAY_URL = 'https://www.gstatic.com/generate_204'

# Nodes that are not real outbound proxies / are informational rows.
SKIP_NAMES = {'DIRECT', 'REJECT', 'REJECT-DROP', 'COMPATIBLE', 'GLOBAL', 'PASS'}
SKIP_PATTERNS = ('套餐', '剩余', '重置', '到期', '官网', '流量')
GROUP_TYPES = {'Selector', 'URLTest', 'Fallback', 'LoadBalance', 'Relay'}
NODE_TYPES = {
    'Shadowsocks', 'ShadowsocksR', 'Vmess', 'Vless', 'Trojan', 'Hysteria',
    'Hysteria2', 'Snell', 'TUIC', 'Socks5', 'Http', 'WireGuard', 'AnyTLS',
}
LOCK_PATH = os.path.join(os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
                         'codex-node-watchdog.lock')
STATE_PATH = os.path.expanduser('~/.cache/codex-node-watchdog.json')


def acquire_lock(path=LOCK_PATH):
    """Single-instance guard: the timer and manual runs must not overlap."""
    handle = open(path, 'a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def log(message):
    print('[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), message), flush=True)


def no_proxy_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Controller(object):
    """Thin client for the mihomo external controller."""

    def __init__(self, base):
        self.base = base.rstrip('/')
        self.opener = no_proxy_opener()

    def proxies(self):
        with self.opener.open(self.base + '/proxies', timeout=15) as response:
            return json.load(response)['proxies']

    def current(self, group):
        return self.proxies()[group].get('now')

    def select(self, group, node):
        body = json.dumps({'name': node}).encode()
        request = urllib.request.Request(
            '%s/proxies/%s' % (self.base, urllib.parse.quote(group)),
            data=body, method='PUT',
            headers={'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=15) as response:
            return response.status

    def delay(self, node, timeout_ms=5000):
        url = '%s/proxies/%s/delay?timeout=%d&url=%s' % (
            self.base, urllib.parse.quote(node), timeout_ms,
            urllib.parse.quote(DELAY_URL, safe=''))
        try:
            with self.opener.open(url, timeout=(timeout_ms / 1000.0) + 15) as response:
                return json.load(response).get('delay')
        except Exception:
            return None


def candidates(proxies, group):
    """All concrete, non-informational nodes of the group, in config order."""
    members = proxies.get(group, {}).get('all') or []
    result = []
    for name in members:
        info = proxies.get(name)
        if not info:
            continue
        if info.get('type') in GROUP_TYPES or info.get('type') not in NODE_TYPES:
            continue
        if name in SKIP_NAMES or any(p in name for p in SKIP_PATTERNS):
            continue
        result.append(name)
    return result


def codex_probe(node, controller, group, host, path, headers, proxy, timeout, samples=1):
    """Pin `node`, then ask the Codex backend.

    Returns ``(verdict, latency_ms, detail)``; ``latency_ms`` is the *fastest* of
    ``samples`` attempts (single samples are far too noisy: the same node has been
    measured anywhere between 0.4s and 16.5s within minutes).
    """
    try:
        controller.select(group, node)
    except Exception as exc:
        return 'dead', None, 'select failed: %s' % exc
    time.sleep(0.5)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({
        'http': proxy, 'https': proxy}))
    latencies = []
    last_error = 'no response'
    for attempt in range(max(1, samples)):
        request = urllib.request.Request(
            host + path, headers=dict(headers, **{'Accept': 'application/json'}))
        started = time.time()
        try:
            with opener.open(request, timeout=timeout) as response:
                latencies.append((time.time() - started) * 1000)
        except urllib.error.HTTPError as exc:
            body = ''
            try:
                body = exc.read().decode('utf-8', 'replace')
            except Exception:
                pass
            if exc.code == 401 and 'invalid_api_key' in body:
                return 'account', None, '401 invalid_api_key (account side, tunnel is fine)'
            return 'blocked', None, 'HTTP %s %s' % (exc.code, ' '.join(body.split())[:80])
        except Exception as exc:
            last_error = type(exc).__name__
        if attempt + 1 < samples:
            time.sleep(0.2)

    if not latencies:
        return 'dead', None, last_error
    best = min(latencies)
    return 'ok', best, '200, fastest %.0fms of %d' % (best, len(latencies))


def load_state(path=STATE_PATH):
    try:
        with open(path) as handle:
            state = json.load(handle)
    except Exception:
        state = {}
    state.setdefault('blocked', {})
    state.setdefault('good', {})
    return state


def save_state(state, path=STATE_PATH):
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(state, handle, ensure_ascii=False, indent=1)
    except Exception as exc:
        log('WARN could not save state %s: %s' % (path, exc))


def order_candidates(alive, state, block_hours):
    """Recently-good nodes first, nodes that answered with a CF block last.

    Probing a candidate means routing live codex traffic through it for a
    moment, so a node that answered 403 (e.g. the Hong Kong one) must not be
    tried again for a while -- that is exactly what breaks the user's turn.
    """
    now = time.time()
    blocked = set(n for n, t in state['blocked'].items() if now - t < block_hours * 3600)
    good = set(n for n, t in state['good'].items() if now - t < 6 * 3600)
    names = [n for _, n in alive]
    ordered = [n for n in names if n in good and n not in blocked]
    ordered += [n for n in names if n not in good and n not in blocked]
    if not ordered:
        ordered = [n for n in names if n not in good] or names
    return ordered


def load_auth(codex_home):
    path = os.path.join(codex_home, 'auth.json')
    with open(path) as handle:
        auth = json.load(handle)
    tokens = auth.get('tokens') or {}
    token, account = tokens.get('access_token'), tokens.get('account_id')
    if not token or not account:
        raise RuntimeError('%s has no ChatGPT tokens (auth_mode=%s)' % (path, auth.get('auth_mode')))
    return {'Authorization': 'Bearer ' + token, 'chatgpt-account-id': account,
            'User-Agent': 'codex_cli_rs/0.155.0', 'originator': 'codex_cli_rs'}


def scan_delays(controller, nodes, workers, timeout_ms):
    alive = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for name, delay in zip(nodes, pool.map(
                lambda n: controller.delay(n, timeout_ms), nodes)):
            if delay:
                alive.append((delay, name))
    alive.sort()
    return alive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--controller', default=DEFAULT_CONTROLLER)
    parser.add_argument('--proxy', default=DEFAULT_PROXY)
    parser.add_argument('--group', default=DEFAULT_GROUP)
    parser.add_argument('--codex-home', default=DEFAULT_CODEX_HOME)
    parser.add_argument('--max-candidates', type=int, default=3,
                        help='candidates to probe (each probe briefly routes live '
                             'traffic through that node, so keep it small)')
    parser.add_argument('--block-hours', type=float, default=6.0,
                        help='how long to avoid re-probing a node that returned a '
                             'Cloudflare block (e.g. the Hong Kong node)')
    parser.add_argument('--probe-timeout', type=float, default=15.0)
    parser.add_argument('--candidate-timeout', type=float, default=10.0,
                        help='a candidate slower than this is useless to codex anyway')
    parser.add_argument('--samples', type=int, default=2,
                        help='probes per node; the fastest sample is kept')
    parser.add_argument('--slow-ms', type=float, default=3000.0,
                        help='treat an answering-but-slow pinned node as unhealthy '
                             '(measured spread across nodes is ~0.4-5s, so 3s keeps '
                             'a fast node without thrashing)')
    parser.add_argument('--improve-ratio', type=float, default=0.5,
                        help='only switch a slow node when a candidate is this much faster')
    parser.add_argument('--retries', type=int, default=1,
                        help='extra attempts on the pinned node before re-pinning '
                             '(the same node often recovers a minute later)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)

    lock = acquire_lock()
    if lock is None:
        log('another watchdog run is in progress (lock %s); skipping' % LOCK_PATH)
        return 0

    controller = Controller(args.controller)
    try:
        proxies = controller.proxies()
    except Exception as exc:
        log('FATAL mihomo controller %s unreachable: %s' % (args.controller, exc))
        return 2
    if args.group not in proxies:
        log('FATAL proxy group %r not found' % args.group)
        return 2

    try:
        headers = load_auth(args.codex_home)
    except Exception as exc:
        log('FATAL %s' % exc)
        return 2

    current = controller.current(args.group)
    verdict, latency, detail = codex_probe(current, controller, args.group, PROBE_HOST,
                                           PROBE_PATH, headers, args.proxy, args.probe_timeout)
    log('pinned node %r -> %s' % (current, detail))

    attempts = 0
    while verdict not in ('ok', 'account') and attempts < args.retries:
        attempts += 1
        log('  attempt %d/%d on the same node (nodes often recover briefly)'
            % (attempts, args.retries))
        time.sleep(3)
        verdict, latency, detail = codex_probe(
            current, controller, args.group, PROBE_HOST, PROBE_PATH,
            headers, args.proxy, args.probe_timeout)
        log('pinned node %r -> %s' % (current, detail))

    report = {'pinned': current, 'pinned_verdict': verdict, 'action': 'none'}
    if verdict == 'account':
        log('WARN Codex backend reachable but the account is rejected -- '
            'not a proxy problem, leaving %r pinned' % current)
        report['action'] = 'none-account'
        if args.json:
            print(json.dumps(report, ensure_ascii=False))
        return 0

    # A node can answer 200 and still be unusably slow (ping says nothing about
    # the Codex backend), so a "healthy" node also gets compared when slow.
    slow = verdict == 'ok' and latency is not None and latency > args.slow_ms
    if verdict == 'ok' and not slow:
        if args.json:
            print(json.dumps(report, ensure_ascii=False))
        return 0

    nodes = candidates(proxies, args.group)
    reason = 'too slow (%.0fms > %dms)' % (latency, args.slow_ms) if slow else 'failed'
    log('pinned node %s, scanning %d candidate(s) by ping' % (reason, len(nodes)))
    alive = scan_delays(controller, nodes, workers=8, timeout_ms=5000)
    log('reachable by ping: %s' % (', '.join('%s(%dms)' % (n, d) for d, n in alive) or 'none'))

    measured = []
    state = load_state()
    ordered = order_candidates(alive, state, args.block_hours)
    log('probe order (recently-good first, CF-blocked skipped): %s'
        % (', '.join(ordered[:args.max_candidates]) or 'none'))
    ping_of = dict((n, d) for d, n in alive)

    for node in ordered[:args.max_candidates]:
        node_verdict, node_latency, node_detail = codex_probe(
            node, controller, args.group, PROBE_HOST, PROBE_PATH,
            headers, args.proxy, args.candidate_timeout, samples=args.samples)
        log('  candidate %-16s ping=%-7s -> %s'
            % (node, '%sms' % ping_of.get(node), node_detail))
        if node_verdict == 'ok':
            measured.append((node_latency, node))
            state['good'][node] = time.time()
        elif node_verdict == 'account':
            measured.append((float('inf'), node))
        elif node_verdict == 'blocked':
            state['blocked'][node] = time.time()
    save_state(state)
    measured.sort()

    if slow and measured:
        best_latency, best_node = measured[0]
        if best_latency < latency * args.improve_ratio:
            log('candidate %r is %.0fms vs %.0fms pinned -- switching'
                % (best_node, best_latency, latency))
        else:
            log('no candidate at least %.1fx faster than %r (%.0fms); keeping it'
                % (args.improve_ratio, current, latency))
            measured = []

    if not measured:
        # Probing candidates moved the group away from the pinned node: undo that.
        try:
            controller.select(args.group, current)
        except Exception:
            pass
        if slow:
            report['action'] = 'kept'
            if args.json:
                print(json.dumps(report, ensure_ascii=False))
            return 0
        log('ERROR no Codex-capable node found; keeping %r' % current)
        report['action'] = 'none-found'
        if args.json:
            print(json.dumps(report, ensure_ascii=False))
        return 1

    best = measured[0][1]
    report['action'] = 'dry-run' if args.dry_run else 'repinned'
    report['new_node'] = best
    if args.dry_run:
        log('DRY-RUN would pin %r -> %r' % (args.group, best))
        try:
            controller.select(args.group, current)
        except Exception:
            pass
    else:
        try:
            controller.select(args.group, best)
        except Exception as exc:
            log('ERROR could not pin %r: %s' % (best, exc))
            return 2
        log('repinned %r: %s -> %s' % (args.group, current, best))
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
