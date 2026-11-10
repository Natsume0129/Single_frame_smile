/* Adds region-only and matched-control responses to the existing clip gallery.
 * Original video, stills, filters, blue/green/red curves and seek coordinates stay
 * on the same reviewed frame/PTS timeline. This file never changes annotations.
 */
(() => {
  'use strict';
  const regions = ['full', 'periocular', 'nose', 'mouth_perioral', 'upper_face', 'lower_face'];
  const regionNames = {full: 'Full face · matched control', periocular: 'Eye region', nose: 'Nose',
    mouth_perioral: 'Mouth / perioral', upper_face: 'Upper face', lower_face: 'Lower face'};
  const typeNames = {polite: 'Polite', true: 'True', bitter: 'Bitter'};
  const palette = {
    full: ['#7c3aed', '#0891b2', '#c026d3'],
    periocular: ['#b45309', '#65a30d', '#db2777'],
    nose: ['#0e7490', '#4f46e5', '#a16207'],
    mouth_perioral: ['#f97316', '#9333ea', '#00857d'],
    upper_face: ['#a21caf', '#64748b', '#be123c'],
    lower_face: ['#6b8e23', '#a85d4b', '#09698c']
  };
  const series = regions.flatMap(region => order.map((label, i) => ({
    key: region + '__' + label, region, label, color: palette[region][i],
    name: regionNames[region] + ' / ' + typeNames[label]
  })));
  const enabled = new Set(series.map(s => s.key));
  const regionalCache = new Map(), pending = new Map(), failures = new Map();
  const panel = document.createElement('section');
  panel.className = 'regional-controls';
  panel.innerHTML = '<div class="bar"><label>Additional region <select id="regionalSelect">'
    + '<option value="none">None — original three models only</option>'
    + regions.map(r => '<option value="' + r + '"' + (r === 'mouth_perioral' ? ' selected' : '') + '>' + regionNames[r] + '</option>').join('')
    + '<option value="all">All regions and matched full-face control</option></select></label>'
    + '<button id="regionalShowAll" type="button">Show selected curves</button>'
    + '<button id="regionalHideAll" type="button">Hide selected curves</button></div>'
    + '<div id="regionalLegend" class="legend regional-legend"></div>'
    + '<div id="regionalStatus" class="small" role="status" aria-live="polite"></div>'
    + '<p class="small">Original full-face curves remain blue (Polite), green (True), and red (Bitter). '
    + 'Additional curves use separately trained region-only models; the matched full-face control uses their same training protocol. '
    + 'Every score uses its own model-specific ten-level reference scale, not a calibrated category probability.</p>';
  const plot = $('chart').parentElement;
  plot.insertBefore(panel, $('chart'));
  const style = document.createElement('style');
  style.textContent = '.regional-controls{margin:10px 0 6px;padding:10px 12px;border:1px solid #d9e3ee;border-radius:8px;background:#f8fafc}.regional-controls .bar{margin:0 0 8px}.regional-legend{gap:7px 16px}.regional-legend label{font-size:12px;font-weight:600;display:flex;align-items:center;gap:3px}.regional-legend input{margin:3px}.regional-controls p{margin:7px 0 0}#regionalStatus.error{color:#a4262c}#regionalCsv{margin-left:10px}';
  document.head.append(style);
  const csvButton = document.createElement('button');
  csvButton.id = 'regionalCsv';
  csvButton.type = 'button';
  csvButton.textContent = 'Download visible curves CSV';
  $('csv').parentElement.append(csvButton);

  function scopeSeries() {
    const selected = $('regionalSelect').value;
    return selected === 'none' ? [] : series.filter(s => selected === 'all' || s.region === selected);
  }
  function selectedSeries() { return scopeSeries().filter(s => enabled.has(s.key)); }
  function currentRegional() { return current && regionalCache.get(current); }
  function status(text, error = false) {
    $('regionalStatus').textContent = text;
    $('regionalStatus').className = 'small' + (error ? ' error' : '');
  }
  function updateLegend() {
    $('regionalLegend').replaceChildren();
    for (const item of scopeSeries()) {
      const label = document.createElement('label');
      label.style.color = item.color;
      const input = document.createElement('input');
      input.type = 'checkbox'; input.checked = enabled.has(item.key); input.dataset.series = item.key;
      input.onchange = () => {input.checked ? enabled.add(item.key) : enabled.delete(item.key); draw(); renderAvailability();};
      label.append(input, document.createTextNode('● ' + item.name));
      $('regionalLegend').append(label);
    }
    draw();
  }
  function validateData(id, payload) {
    const record = byId[id];
    if (!record || !payload || !Array.isArray(payload.points) || payload.points.length !== record.tracked_frames) {
      throw new Error('Unexpected regional clip identity or frame count.');
    }
    if (payload.id !== undefined && payload.id !== id) throw new Error('Regional clip ID mismatch.');
    if (payload.person !== undefined && payload.person !== record.person) throw new Error('Regional person mismatch.');
    if (payload.label !== undefined && payload.label !== record.label) throw new Error('Regional category mismatch.');
    const points = payload.points;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (!Number.isFinite(p.seconds) || !Number.isInteger(p.source_frame) || (i && (p.seconds <= points[i-1].seconds || p.source_frame <= points[i-1].source_frame))) {
        throw new Error('Regional time/frame order is invalid.');
      }
      for (const s of series) for (const suffix of ['', '_smooth']) {
        const v = p[s.key + suffix];
        if (!Number.isFinite(v) || v < -1e-6 || v > 9 + 1e-6) throw new Error('Regional score missing or out of range: ' + s.key + suffix);
      }
    }
    if (Math.abs(points[0].seconds) > 1e-5 || Math.abs(points[points.length-1].seconds - record.interval_seconds) > 1e-5) {
      throw new Error('Regional interval does not match the reviewed clip.');
    }
    return payload;
  }
  function aligned(payload) {
    return clipData && payload.points.length === clipData.points.length && payload.points.every((p,i) =>
      p.source_frame === clipData.points[i].source_frame && Math.abs(p.seconds - clipData.points[i].seconds) < 1e-6);
  }
  function renderAvailability() {
    if (!current) return;
    csvButton.disabled = !clipData || (selectedSeries().length > 0 && !currentRegional());
    if ($('regionalSelect').value === 'none') {status('Showing the original three full-face models.'); return;}
    if (failures.has(current)) {status(failures.get(current) + ' Original curves and video remain available.', true); return;}
    if (!currentRegional()) {status('Loading regional curves…'); return;}
    if (clipData && !aligned(currentRegional())) {
      status('Regional and original frames do not match. Regional curves are hidden.', true); csvButton.disabled = true; return;
    }
    status('Regional curves loaded · same ' + byId[current].tracked_frames + ' tracked frames and reviewed onset → peak interval.');
  }
  function loadRegional() {
    if (!current) return;
    const id = current, record = byId[id];
    if (regionalCache.has(id) || pending.has(id) || failures.has(id)) {renderAvailability(); return;}
    if (!record.regional_curve_js) {failures.set(id, 'No regional curve file is registered for this clip.'); renderAvailability(); return;}
    const script = document.createElement('script');
    script.src = record.regional_curve_js;
    pending.set(id, script);
    script.onload = () => {
      pending.delete(id); script.remove();
      if (!regionalCache.has(id) && !failures.has(id)) failures.set(id, 'Regional file loaded without a valid data callback.');
      if (current === id) {renderAvailability(); draw();}
    };
    script.onerror = () => {
      pending.delete(id); script.remove(); failures.set(id, 'Could not load regional curves. Keep both output folders in their current locations.');
      if (current === id) {renderAvailability(); draw();}
    };
    document.body.append(script); renderAvailability();
  }
  window.showRegionalClipData = (id, payload) => {
    try {regionalCache.set(id, validateData(id, payload)); failures.delete(id);}
    catch (error) {regionalCache.delete(id); failures.set(id, error.message);}
    if (id === current) {renderAvailability(); draw();}
  };

  const baselineShow = show;
  show = function() {baselineShow(); loadRegional();};
  const baselineApplyData = applyData;
  applyData = function(data) {baselineApplyData(data); renderAvailability();};
  const baselineDraw = draw;
  draw = function(cursor = true) {
    const chosen = selectedSeries(), c = $('chart');
    // The original coordinate system stays fixed; extra printable legend occupies
    // additional canvas space below it, so video seeking keeps exact alignment.
    const height = 430 + Math.ceil(chosen.length / 3) * 24;
    if (c.height !== height) {c.height = height; c.style.height = Math.round(height * 360 / 430) + 'px';}
    baselineDraw(cursor);
    if (!current || !clipData) return;
    const payload = currentRegional();
    if (!payload || !aligned(payload)) return;
    const ctx = c.getContext('2d'), end = Math.max(byId[current].interval_seconds, .001);
    const x = t => 70 + 980 * t / end, y = value => 350 - 300 * value / 9;
    const suffix = $('display').value === 'smooth' ? '_smooth' : '';
    chosen.forEach((item, index) => {
      ctx.strokeStyle = ctx.fillStyle = item.color; ctx.lineWidth = 1.7;
      ctx.setLineDash(item.region === 'full' ? [7, 3] : []); ctx.beginPath();
      payload.points.forEach((p, i) => {
        if (!i || p.seconds - payload.points[i-1].seconds > .15) ctx.moveTo(x(p.seconds), y(p[item.key + suffix]));
        else ctx.lineTo(x(p.seconds), y(p[item.key + suffix]));
      });
      ctx.stroke(); ctx.setLineDash([]);
      for (const p of payload.points) {ctx.beginPath(); ctx.arc(x(p.seconds), y(p[item.key + suffix]), 2.4, 0, Math.PI * 2); ctx.fill();}
      ctx.font = '12px Arial'; ctx.fillText('● ' + item.name, 70 + (index % 3) * 330, 446 + Math.floor(index / 3) * 24);
    });
    if (cursor) {ctx.strokeStyle = '#34495e'; ctx.lineWidth = 1; ctx.setLineDash([5,4]); const position = x(Math.max(0,Math.min(end,$('video').currentTime))); ctx.beginPath(); ctx.moveTo(position,45); ctx.lineTo(position,350); ctx.stroke(); ctx.setLineDash([]);}
  };
  const baselineHover = $('chart').onmousemove;
  $('chart').onmousemove = event => {
    baselineHover(event);
    const payload = currentRegional();
    if (!payload || !aligned(payload)) return;
    const time = timeAt(event), p = payload.points.reduce((a,b) => Math.abs(a.seconds-time) <= Math.abs(b.seconds-time) ? a : b);
    const suffix = $('display').value === 'smooth' ? '_smooth' : '';
    for (const item of selectedSeries()) $('tooltip').textContent += ' · ' + item.name + ' ' + p[item.key + suffix].toFixed(3);
  };
  $('regionalSelect').onchange = () => {updateLegend(); loadRegional(); renderAvailability();};
  $('regionalShowAll').onclick = () => {scopeSeries().forEach(s => enabled.add(s.key)); updateLegend(); renderAvailability();};
  $('regionalHideAll').onclick = () => {scopeSeries().forEach(s => enabled.delete(s.key)); updateLegend(); renderAvailability();};
  csvButton.onclick = () => {
    if (!current || !clipData) return;
    const chosen = selectedSeries(), payload = currentRegional();
    if (chosen.length && (!payload || !aligned(payload))) {renderAvailability(); return;}
    const original = order.filter(label => $(label).checked), suffix = $('display').value === 'smooth' ? '_smooth' : '';
    const columns = ['seconds', 'source_seconds', 'source_frame', ...original.map(l => 'original__' + l + suffix), ...chosen.map(s => s.key + suffix)];
    const rows = clipData.points.map((p,i) => [p.seconds,p.source_seconds,p.source_frame,...original.map(l => p[l + suffix]),...chosen.map(s => payload.points[i][s.key + suffix])].join(','));
    const blob = new Blob([columns.join(',') + '\r\n' + rows.join('\r\n') + '\r\n'], {type:'text/csv;charset=utf-8'});
    const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url;
    a.download = byId[current].person + '_' + byId[current].label + '_' + current + '_visible_curves.csv'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  // Keep the original download button, but include the new visible legend/curves.
  $('savePlot').onclick = () => {if(!current)return; draw(false); const a=document.createElement('a'); a.href=$('chart').toDataURL('image/png'); a.download=byId[current].person+'_'+byId[current].label+'_'+current+'_regional_models.png'; a.click(); draw();};
  updateLegend(); loadRegional(); renderAvailability();
})();
