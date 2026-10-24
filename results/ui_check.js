
const $ = (id) => document.getElementById(id);
async function refresh() {
  try {
    const r = await fetch('/health'); const j = await r.json();
    $('status').textContent = `budget ${j.budget_gb}GB | bos VRAM ${j.free_vram_gb}GB | ` +
      `ckpt ${j.deep_ckpt || 'base'} | istek ${j.counters.requests} | cache ${j.cache.hits}`;
  } catch (e) { $('status').textContent = 'servis yok'; }
}
async function preload() {
  $('pre').disabled = true; $('status').textContent = 'deep yukleniyor...';
  try {
    const r = await fetch('/v1/preload', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({units: ['deep']})});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    $('status').textContent = `deep hazir (${j.elapsed_s}s, bos VRAM ${j.free_vram_gb}GB)`;
  } catch (e) {
    $('status').textContent = 'preload hatasi: ' + e.message;
  } finally { $('pre').disabled = false; }
}
function showMeta(j) {
  const plan = j.plan.map(g => g.units.join(' + ') + ' [' + g.kind + ', ' + g.wall_s + 's]').join('  ->  ');
  const units = Object.entries(j.per_unit).map(([k, v]) =>
    `${k}: ${v.gen_s}s` + (v.score !== null ? `, skor ${v.score}` : '')).join(' | ');
  $('meta').textContent =
    `rota: ${j.route.reason}\nplan (${j.n_groups} grup${j.parallel ? ', paralel' : ''}): ${plan}\n` +
    `birimler: ${units}\ntoplam ${j.elapsed_s}s (yukleme ${j.timings.load_s}s, ` +
    `kuyruk ${j.timings.queue_s || 0}s, uretim ${j.timings.gen_s}s) | ` +
    `peak VRAM ${j.peak_vram_mb}MB | merge=${j.merge} | istek #${j.request_id}` +
    (j.cached ? ` (cache; ilk ${j.cached_orig_elapsed_s}s)` : '') +
    (j.warnings && j.warnings.length ? `\nikaz: ${j.warnings.join(' | ')}` : '');
}
async function ask() {
  const q = $('q').value.trim(); if (!q) return;
  const body = {query: q, mode: $('mode').value, merge: $('merge').value,
                max_new_tokens: Number($('n').value)};
  $('go').disabled = true; $('status').textContent = 'calisiyor...';
  $('answer').textContent = '...'; $('meta').textContent = '';
  try {
    if (!$('stream').checked) {
      const r = await fetch('/v1/answer', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      $('answer').textContent = j.answer || '(bos)';
      showMeta(j);
    } else {
      const r = await fetch('/v1/stream', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      if (!r.ok || !(r.headers.get('content-type') || '').includes('text/event-stream')) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error || r.statusText);
      }
      const reader = r.body.getReader(); const dec = new TextDecoder();
      let buf = '', text = '', final = null;
      $('answer').textContent = '';
      while (true) {
        const {value, done} = await reader.read(); if (done) break;
        buf += dec.decode(value, {stream: true});
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
          if (!block.startsWith('data: ')) continue;
          const ev = JSON.parse(block.slice(6));
          if (ev.event === 'token') {
            text += ev.text; $('answer').textContent = text;
          } else if (ev.event === 'final') {
            final = ev.response;
          } else if (ev.event === 'error') {
            throw new Error(ev.error);
          }
        }
      }
      if (final) { $('answer').textContent = final.answer || text; showMeta(final); }
    }
  } catch (e) {
    $('answer').textContent = 'Hata: ' + e.message;
  } finally { $('go').disabled = false; refresh(); }
}
$('go').onclick = ask;
$('pre').onclick = preload;
document.querySelectorAll('.ex').forEach((b) => {
  b.onclick = () => { $('q').value = b.dataset.q; $('q').focus(); };
});
$('q').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask(); });
refresh();
