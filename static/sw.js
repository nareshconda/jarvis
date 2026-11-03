const CACHE = 'jarvis-v1';                 // bump this string to push an update
const PRECACHE = ['/static/icons/icon-192.png', '/static/icons/icon-512.png'];

self.addEventListener('install', e => {
    self.skipWaiting();
    e.waitUntil(caches.open(CACHE).then(c => c.addAll(PRECACHE)).catch(() => {}));
});

self.addEventListener('activate', e => {
    e.waitUntil(
        caches.keys()
        .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
        .then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', e => {
    const req = e.request;
    if (req.method !== 'GET') return;                       // leave POST / uploads alone
    const url = new URL(req.url);
    if (url.origin !== self.location.origin) return;        // ignore Google / Cloudflare domains
    if (url.pathname.startsWith('/api/') ||
        url.pathname.startsWith('/files/')) return;         // always live, never cached

        if (url.pathname.startsWith('/static/')) {              // assets: cache-first
            e.respondWith(
                caches.match(req).then(hit => hit || fetch(req).then(res => {
                    if (res.ok && res.type === 'basic') caches.open(CACHE).then(c => c.put(req, res.clone()));
                    return res;
                }))
            );
            return;
        }

        // everything else (the app page): network-first, fall back to cache if offline
        e.respondWith(
            fetch(req).then(res => {
                if (res.ok && res.type === 'basic') caches.open(CACHE).then(c => c.put(req, res.clone()));
                return res;
            }).catch(() => caches.match(req))
        );
});
