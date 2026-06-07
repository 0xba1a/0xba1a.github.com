/* ============================================================================
   Generic click-through step machine for Shorts.
   - One `.scene` per narration line; advance with click / ArrowRight / Space.
   - Per-scene entrance animations: define window.SCENE_HOOKS = { 1: fn, 2: fn }
     and each fn(sceneEl) runs when that scene becomes active.
   The builder drives this headlessly, holding each scene for its narration
   clip length, so keep sub-animations shorter than the spoken line.
   ============================================================================ */
(() => {
  const stage = document.getElementById('stage');
  const scenes = Array.from(document.querySelectorAll('.scene'));
  const stepCounter = document.getElementById('stepCounter');
  const clickHint = document.getElementById('clickHint');
  const TOTAL = scenes.length;
  const hooks = window.SCENE_HOOKS || {};
  let current = 0;

  function updateCounter() {
    if (stepCounter) stepCounter.textContent = `${current} / ${TOTAL}`;
  }

  function show(idx) {
    scenes.forEach(s => s.classList.remove('active'));
    const el = scenes[idx - 1];
    if (!el) return;
    el.classList.add('active');
    const fn = hooks[idx];
    if (typeof fn === 'function') {
      // let the opacity transition start, then run the entrance animation
      requestAnimationFrame(() => fn(el));
    }
  }

  function advance() {
    if (current >= TOTAL) return;
    current++;
    show(current);
    updateCounter();
    if (current >= TOTAL && clickHint) clickHint.textContent = '← Tap to restart';
  }

  function reset() {
    current = 0;
    scenes.forEach(s => s.classList.remove('active'));
    if (clickHint) clickHint.textContent = 'Tap or press → to advance';
    updateCounter();
  }

  let last = 0;
  function handle() {
    const now = Date.now();
    if (now - last < 250) return;      // debounce
    last = now;
    if (current >= TOTAL) reset(); else advance();
  }

  stage.addEventListener('click', handle);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); handle(); }
    if (e.key === 'ArrowLeft' || e.key === 'r') reset();
  });

  updateCounter();
})();
