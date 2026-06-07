(() => {
  const stage = document.getElementById('stage');
  const stepCounter = document.getElementById('stepCounter');
  const clickHint = document.getElementById('clickHint');

  const sceneHook     = document.querySelector('.scene-hook');
  const sceneGrid     = document.querySelector('.scene-grid');
  const sceneLeap     = document.querySelector('.scene-leap');
  const sceneSsd      = document.querySelector('.scene-ssd');
  const sceneCompare  = document.querySelector('.scene-compare');
  const sceneTakeaway = document.querySelector('.scene-takeaway');
  const allScenes = [sceneHook, sceneGrid, sceneLeap, sceneSsd, sceneCompare, sceneTakeaway];

  const TOTAL_STEPS = 8;
  let currentStep = 0;
  let initedGrid = false;
  let initedLeap = false;
  let initedSsd  = false;
  let leapTickInterval = null;
  let ssdRampTransition = null;

  /* ════════════════════════════════════════════
     Utility
     ════════════════════════════════════════════ */
  function showScene(el) {
    allScenes.forEach(s => s.classList.remove('active'));
    el.classList.add('active');
  }
  function shake(ms = 500) {
    stage.classList.remove('shake');
    void stage.offsetWidth;
    stage.classList.add('shake');
    setTimeout(() => stage.classList.remove('shake'), ms);
  }

  /* ════════════════════════════════════════════
     Scene 2 + 3 — Hardware grid
     ════════════════════════════════════════════ */
  const GRID_COLS = 8;
  const GRID_ROWS = 10;
  const GRID_TOTAL = GRID_COLS * GRID_ROWS;
  let gridCells = null;
  let randomFailIndices = [];

  function initGrid() {
    if (initedGrid) return;
    initedGrid = true;

    const container = document.getElementById('gridContainer');
    const W = 800, H = 1000;
    const padX = 40, padY = 30;
    const cw = (W - padX * 2) / GRID_COLS;
    const ch = (H - padY * 2) / GRID_ROWS;

    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const data = d3.range(GRID_TOTAL).map(i => ({
      i,
      col: i % GRID_COLS,
      row: Math.floor(i / GRID_COLS)
    }));

    const cellG = svg.selectAll('g.cell')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'cell')
      .attr('transform', d => `translate(${padX + d.col * cw},${padY + d.row * ch})`)
      .style('opacity', 0);

    // Disk body (rounded rect)
    cellG.append('rect')
      .attr('class', 'cell-body')
      .attr('width', cw - 8)
      .attr('height', ch - 10)
      .attr('rx', 6);

    // Status LED (small circle)
    cellG.append('circle')
      .attr('class', 'cell-led')
      .attr('cx', (cw - 8) - 10)
      .attr('cy', 10)
      .attr('r', 4);

    // Decorative platter line
    cellG.append('line')
      .attr('x1', 8).attr('x2', cw - 16)
      .attr('y1', (ch - 10) / 2).attr('y2', (ch - 10) / 2)
      .attr('stroke', '#30363d')
      .attr('stroke-width', 1);

    // Stagger fade-in
    cellG.transition()
      .delay((d, i) => i * 8)
      .duration(300)
      .style('opacity', 1);

    gridCells = cellG;

    // Pre-pick 6 random failure indices (deterministic seed)
    randomFailIndices = pickN(GRID_TOTAL, 6, 1337);
  }

  function pickN(total, n, seed) {
    const out = [];
    let s = seed;
    const taken = new Set();
    while (out.length < n) {
      s = (s * 16807) % 2147483647;
      const idx = s % total;
      if (!taken.has(idx)) {
        taken.add(idx);
        out.push(idx);
      }
    }
    return out;
  }

  function runRandomFailures() {
    const counterEl = document.getElementById('gridCounter');
    counterEl.classList.add('visible');
    counterEl.textContent = 'Failed: 0';

    randomFailIndices.forEach((failIdx, k) => {
      setTimeout(() => {
        gridCells.filter(d => d.i === failIdx)
          .select('.cell-led')
          .classed('failing', true)
          .transition().duration(180)
          .attr('r', 6)
          .transition().duration(180)
          .attr('r', 4);

        gridCells.filter(d => d.i === failIdx)
          .select('.cell-body')
          .transition().duration(280)
          .attr('stroke', '#f85149')
          .style('opacity', .55);

        counterEl.textContent = `Failed: ${k + 1}`;
      }, 600 + k * 1100);
    });
  }

  function runSimultaneousFailure() {
    // Restore any previously failed cells visually first (in case of replay)
    document.getElementById('gridCounter').classList.remove('visible');
    document.getElementById('gridBanner').classList.add('visible');

    gridCells.select('.cell-led')
      .interrupt()
      .classed('failing', true);

    gridCells.select('.cell-body')
      .interrupt()
      .transition().duration(120)
      .attr('stroke', '#f85149')
      .style('opacity', .85);

    shake(550);
  }

  /* ════════════════════════════════════════════
     Scene 4 — Leap second clock + world map
     ════════════════════════════════════════════ */
  // Server-cluster hubs — each spawns a swarm of nearby dots, simulating
  // datacenters / PoPs / colocated server farms across the globe.
  const CITY_HUBS = [
    // North America
    [-77.0,  38.9], [-122.0, 37.4], [-122.4, 37.78], [-118.2, 34.05],
    [ -73.9, 40.7], [ -87.6, 41.8], [ -95.3, 29.7], [ -97.7, 30.3],
    [ -84.4, 33.7], [ -80.2, 25.8], [ -71.0, 42.4], [ -75.2, 39.95],
    [-104.9, 39.7], [-122.3, 47.6], [-123.1, 49.3], [ -79.4, 43.7],
    [ -73.6, 45.5], [ -99.1, 19.4],
    // South America
    [ -46.6,-23.5], [ -58.4,-34.6], [ -70.6,-33.4], [ -74.1,  4.7],
    [ -77.0,-12.0], [ -47.9,-15.8],
    // Europe
    [  -0.1, 51.5], [   2.35,48.85], [   8.7, 50.1], [   4.9, 52.3],
    [  13.4, 52.5], [  12.5, 41.9], [  -3.7, 40.4], [  -9.1, 38.7],
    [  18.1, 59.3], [  10.7, 59.9], [  12.5, 55.7], [  19.0, 47.5],
    [  14.4, 50.1], [  21.0, 52.2], [  37.6, 55.8], [  30.5, 50.4],
    [  28.9, 41.0], [  23.7, 37.98],
    // Africa
    [  31.2, 30.0], [   3.4,  6.5], [  28.0,-26.2], [  18.4,-33.9],
    [  36.8, -1.3], [  39.3, -6.8], [  -7.6, 33.6],
    // Middle East
    [  55.3, 25.2], [  46.7, 24.7], [  44.3, 33.3], [  35.2, 31.8],
    [  51.4, 35.7],
    // South & Central Asia
    [  77.2, 28.6], [  72.9, 19.1], [  77.6, 12.97], [  88.4, 22.6],
    [  80.3, 13.1], [  73.1, 33.7], [  90.4, 23.8], [ 100.5, 13.75],
    [ 101.7,  3.2], [ 106.8, -6.2], [ 121.0, 14.6], [ 106.7, 10.8],
    // East Asia
    [ 116.4, 39.9], [ 121.5, 31.2], [ 113.3, 23.1], [ 114.2, 22.3],
    [ 121.6, 25.0], [ 126.98, 37.57], [ 139.7, 35.7], [ 135.5, 34.7],
    // Oceania
    [ 151.2,-33.8], [ 144.96,-37.8], [ 153.0,-27.5], [ 174.8,-36.85],
    // Russia / North Asia
    [  82.9, 55.0], [  92.9, 56.0], [ 131.9, 43.1],
  ];

  // Build the dot list deterministically: each hub spawns 2–5 nearby dots.
  function buildCities() {
    const out = [];
    let s = 91237;
    function rnd() { s = (s * 16807) % 2147483647; return s / 2147483647; }
    CITY_HUBS.forEach((hub, hi) => {
      // Anchor dot for the hub itself
      out.push({ id: out.length, lonlat: hub });
      const swarmN = 2 + Math.floor(rnd() * 4); // 2–5 satellites
      for (let k = 0; k < swarmN; k++) {
        const dLon = (rnd() - 0.5) * 6.5;  // ±3.25°
        const dLat = (rnd() - 0.5) * 4.5;  // ±2.25°
        out.push({ id: out.length, lonlat: [hub[0] + dLon, hub[1] + dLat] });
      }
    });
    return out;
  }
  const CITIES = buildCities();
  // Pre-pick 70% to fail at the leap second (deterministic).
  const FAIL_RATIO = 0.70;
  const FAIL_SET = (() => {
    const idx = CITIES.map(c => c.id);
    let s = 271828;
    for (let i = idx.length - 1; i > 0; i--) {
      s = (s * 16807) % 2147483647;
      const j = s % (i + 1);
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return new Set(idx.slice(0, Math.floor(idx.length * FAIL_RATIO)));
  })();

  let mapSvg = null, projection = null, cityG = null;

  function initLeapMap() {
    if (initedLeap) return;
    initedLeap = true;

    const mapContainer = document.getElementById('leapMap');
    const W = 800, H = 480;

    mapSvg = d3.select(mapContainer).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    projection = d3.geoNaturalEarth1()
      .scale(155)
      .translate([W / 2, H / 2 + 10]);

    const path = d3.geoPath(projection);
    const mapG = mapSvg.append('g').attr('class', 'map-g');

    // Try to fetch world atlas (TopoJSON). Fall back gracefully.
    const url = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json';
    fetch(url)
      .then(r => r.ok ? r.json() : Promise.reject('map fetch failed'))
      .then(world => {
        const countries = topojson.feature(world, world.objects.countries);
        mapG.selectAll('path.country-path')
          .data(countries.features)
          .enter().append('path')
          .attr('class', 'country-path')
          .attr('d', path);
        drawCities();
      })
      .catch(() => {
        // Fallback: simple equator/meridian frame so the scene still works offline
        mapG.append('rect')
          .attr('x', 1).attr('y', 1)
          .attr('width', W - 2).attr('height', H - 2)
          .attr('fill', '#161b22')
          .attr('stroke', '#30363d');
        mapG.append('line')
          .attr('x1', 0).attr('x2', W)
          .attr('y1', H / 2).attr('y2', H / 2)
          .attr('stroke', '#30363d').attr('stroke-dasharray', '4 4');
        drawCities();
      });
  }

  function drawCities() {
    cityG = mapSvg.append('g').attr('class', 'cities');
    cityG.selectAll('circle.city-dot')
      .data(CITIES)
      .enter().append('circle')
      .attr('class', 'city-dot')
      .attr('cx', d => projection(d.lonlat)[0])
      .attr('cy', d => projection(d.lonlat)[1])
      .attr('r', 0)
      .transition().duration(280)
      .delay((d, i) => Math.min(i * 4, 1400))
      .attr('r', 2.6);
  }

  function runLeapSecond() {
    const clockEl = document.getElementById('leapClock');
    const captionEl = document.getElementById('leapCaption');
    captionEl.classList.remove('visible');
    clockEl.classList.remove('alert');
    clockEl.textContent = '23:59:55';

    let sec = 55;
    if (leapTickInterval) clearInterval(leapTickInterval);

    leapTickInterval = setInterval(() => {
      sec++;
      if (sec <= 59) {
        clockEl.textContent = `23:59:${String(sec).padStart(2, '0')}`;
      } else if (sec === 60) {
        // Leap second
        clockEl.textContent = '23:59:60';
        clockEl.classList.add('alert');
        cascadeCityFailure();
        captionEl.classList.add('visible');
      } else {
        clearInterval(leapTickInterval);
        leapTickInterval = null;
      }
    }, 700);
  }

  function cascadeCityFailure() {
    if (!cityG) return;
    // Fail only the pre-picked ~70%; the remaining 30% stay green.
    cityG.selectAll('circle.city-dot')
      .filter(d => FAIL_SET.has(d.id))
      .transition()
      .delay((d, i) => 40 + (i % 60) * 8)   // quick scatter; total ~500ms
      .duration(140)
      .attr('class', 'city-dot down');

    // Sample ~16 hub-anchor ripples (anchors are even-indexed) to keep it cheap.
    const rippleHubs = CITY_HUBS.filter((_, i) => i % 5 === 0);
    rippleHubs.forEach((hub, i) => {
      const [cx, cy] = projection(hub);
      mapSvg.append('circle')
        .attr('class', 'city-ripple')
        .attr('cx', cx).attr('cy', cy).attr('r', 4)
        .style('opacity', .9)
        .transition()
        .delay(40 + i * 35)
        .duration(900)
        .attr('r', 30)
        .style('opacity', 0)
        .remove();
    });

    setTimeout(() => shake(450), 200);
  }

  /* ════════════════════════════════════════════
     Scene 5 + 6 — SSD overflow
     ════════════════════════════════════════════ */
  let ssdSvg = null;
  let bitsContainer = null;
  let bitEls = [];

  function initSsd() {
    if (initedSsd) return;
    initedSsd = true;

    const stageEl = document.getElementById('ssdStage');
    const W = 800, H = 720;

    ssdSvg = d3.select(stageEl).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // Single, large SSD — fills most of the stage
    const BW = 600, BH = 460;
    const single = ssdSvg.append('g').attr('class', 'ssd-single')
      .attr('transform', `translate(${(W - BW) / 2},${(H - BH) / 2})`);

    single.append('rect')
      .attr('class', 'ssd-body')
      .attr('width', BW).attr('height', BH)
      .attr('rx', 28);
    single.append('circle')
      .attr('class', 'ssd-led')
      .attr('cx', 44).attr('cy', 44).attr('r', 12);
    single.append('text')
      .attr('class', 'ssd-label')
      .attr('x', BW / 2).attr('y', BH / 2 + 10)
      .attr('text-anchor', 'middle')
      .style('font-size', '180px')
      .style('font-weight', '800')
      .style('letter-spacing', '8px')
      .text('SSD');

    // Progress bar near the bottom of the SSD
    const PB_X = 70, PB_Y = BH - 60, PB_W = BW - 140, PB_H = 30;
    single.append('rect')
      .attr('x', PB_X).attr('y', PB_Y)
      .attr('width', PB_W).attr('height', PB_H)
      .attr('rx', 6)
      .attr('fill', '#0d1117').attr('stroke', '#30363d').attr('stroke-width', 1);
    single.append('rect')
      .attr('class', 'ssd-progress')
      .attr('x', PB_X + 2).attr('y', PB_Y + 2)
      .attr('width', 0).attr('height', PB_H - 4)
      .attr('rx', 4)
      .attr('fill', '#58a6ff');
    single.datum({ BW, BH, PB_X, PB_Y, PB_W, PB_H });

    // BRICKED stamp (hidden until overflow)
    single.append('text')
      .attr('class', 'ssd-bricked-text')
      .attr('x', BW / 2).attr('y', BH / 2 + 78)
      .attr('text-anchor', 'middle')
      .style('font-size', '46px')
      .style('letter-spacing', '6px')
      .style('opacity', 0)
      .text('BRICKED');

    // Bits (16)
    bitsContainer = document.getElementById('ssdBits');
    bitsContainer.innerHTML = '';
    for (let b = 15; b >= 0; b--) {
      const bitEl = document.createElement('div');
      bitEl.className = 'bit';
      bitEl.dataset.bit = b;
      bitEl.textContent = '0';
      bitsContainer.appendChild(bitEl);
    }
    bitEls = Array.from(bitsContainer.children);
  }

  function setBits(value16) {
    // value16 may be negative (after overflow). Mask to 16 bits unsigned.
    const v = value16 & 0xFFFF;
    for (let b = 15; b >= 0; b--) {
      const on = (v >> b) & 1;
      const el = bitEls[15 - b]; // first child shows MSB (bit 15)
      el.textContent = on ? '1' : '0';
      el.classList.toggle('on', !!on);
    }
  }

  function runSsdRamp() {
    // Reset state
    document.getElementById('ssdCounter').classList.remove('alert');
    document.getElementById('ssdCaption').classList.remove('alert');
    document.getElementById('ssdCaption').textContent = 'int16 — max value 32,767';
    bitEls.forEach(b => {
      b.classList.remove('sign-flip', 'sign-warn', 'on', 'overflow');
      b.textContent = '0';
    });

    // Restore single SSD if it was bricked during a prior overflow run
    const singleSel = ssdSvg.select('g.ssd-single');
    singleSel.select('rect.ssd-body').classed('bricked', false);
    singleSel.select('circle.ssd-led').classed('bricked', false);
    singleSel.select('text.ssd-label').classed('bricked', false);
    singleSel.select('text.ssd-bricked-text').style('opacity', 0);
    ssdSvg.select('rect.ssd-progress').attr('width', 0).attr('fill', '#58a6ff');

    const counterEl = document.getElementById('ssdCounter');
    const TARGET = 32767;
    const DURATION = 4500;
    const PB_W = singleSel.datum().PB_W;

    if (ssdRampTransition) ssdRampTransition.interrupt();

    ssdRampTransition = ssdSvg.select('rect.ssd-progress')
      .interrupt()
      .transition()
      .duration(DURATION)
      .ease(d3.easeQuadOut)
      .attr('width', PB_W - 4)
      .tween('counter', function () {
        const interp = d3.interpolateNumber(0, TARGET);
        return (t) => {
          const v = Math.floor(interp(t));
          counterEl.textContent = v.toLocaleString() + ' hrs';
          setBits(v);
          // Warn the sign bit when we approach the threshold
          if (v > 30000 && v < 32767) {
            bitEls[0].classList.add('sign-warn');
          }
        };
      })
      .on('end', () => {
        // Force exact final state
        setBits(TARGET);
        counterEl.textContent = TARGET.toLocaleString() + ' hrs';

        // Counter at the limit — turn it red
        counterEl.classList.add('alert');

        // Progress bar — full → red
        ssdSvg.select('rect.ssd-progress')
          .transition().duration(260)
          .attr('fill', '#f85149');

        // Every '1' bit (and only those) flips blue → red.
        // Bit 15 (sign) stays in its yellow warn state until the next click.
        bitEls.forEach((el, idx) => {
          if (idx === 0) return; // skip sign bit; keep its sign-warn glow
          if (el.classList.contains('on')) {
            el.classList.add('overflow');
          }
        });
      });
  }

  function runSsdOverflow() {
    const counterEl = document.getElementById('ssdCounter');
    const captionEl = document.getElementById('ssdCaption');

    if (ssdRampTransition) ssdRampTransition.interrupt();

    // Force final ramp state
    setBits(32767);
    counterEl.textContent = '32,767 hrs';
    bitEls[0].classList.add('sign-warn');

    // === OVERFLOW · everything blue → red, all at once ===
    setTimeout(() => {
      // 1) Bits: every previously-on bit + the sign bit go red simultaneously.
      //    Keep their '1' display for a beat so the "blue ones became red ones"
      //    transition reads clearly.
      bitEls[0].classList.remove('sign-warn');
      bitEls.forEach(el => el.classList.add('overflow'));
      bitEls[0].classList.add('sign-flip', 'on');
      bitEls[0].textContent = '1';

      // 2) Counter + caption flip red.
      counterEl.textContent = '−32,768 hrs';
      counterEl.classList.add('alert');
      captionEl.textContent = 'OVERFLOW · sign bit flipped';
      captionEl.classList.add('alert');

      // 3) The disk: body, LED, label, progress bar — all red in the same beat.
      const singleSel = ssdSvg.select('g.ssd-single');
      singleSel.select('rect.ssd-body')
        .transition().duration(180)
        .attr('fill', '#5a1e1e').attr('stroke', '#f85149');
      singleSel.select('circle.ssd-led').classed('bricked', true);
      singleSel.select('text.ssd-label').classed('bricked', true);
      singleSel.select('rect.ssd-progress')
        .transition().duration(180)
        .attr('fill', '#f85149');
      singleSel.select('text.ssd-bricked-text')
        .transition().delay(140).duration(240)
        .style('opacity', 1);

      // 4) Whole-scene red flash + screen shake.
      sceneSsd.classList.add('flash-red');
      setTimeout(() => sceneSsd.classList.remove('flash-red'), 480);
      shake(500);

      // 5) After the flash, snap bits to the technically-correct −32,768 layout
      //    (sign=1, others=0). They stay red the whole time.
      setTimeout(() => {
        bitEls.forEach((el, i) => {
          el.textContent = i === 0 ? '1' : '0';
        });
      }, 600);
    }, 600);
  }

  /* ════════════════════════════════════════════
     Step machine
     ════════════════════════════════════════════ */
  function step1() { showScene(sceneHook); }

  function step2() {
    showScene(sceneGrid);
    document.getElementById('gridBanner').classList.remove('visible');
    initGrid();
    setTimeout(runRandomFailures, 350);
  }

  function step3() {
    // Same scene; trigger simultaneous failure overlay
    runSimultaneousFailure();
  }

  function step4() {
    showScene(sceneLeap);
    initLeapMap();
    setTimeout(runLeapSecond, 400);
  }

  function step5() {
    showScene(sceneSsd);
    initSsd();
    setTimeout(runSsdRamp, 350);
  }

  function step6() { runSsdOverflow(); }

  function step7() { showScene(sceneCompare); }

  function step8() { showScene(sceneTakeaway); }

  const steps = [null, step1, step2, step3, step4, step5, step6, step7, step8];

  function updateCounter() {
    stepCounter.textContent = `${currentStep} / ${TOTAL_STEPS}`;
  }

  function advance() {
    if (currentStep >= TOTAL_STEPS) return;
    currentStep++;
    steps[currentStep]();
    updateCounter();
    if (currentStep >= TOTAL_STEPS) {
      clickHint.textContent = '← Tap to restart';
    }
  }

  function reset() {
    currentStep = 0;

    // Stop any active timers / transitions
    if (leapTickInterval) { clearInterval(leapTickInterval); leapTickInterval = null; }
    if (ssdRampTransition) { ssdRampTransition.interrupt(); ssdRampTransition = null; }

    // Hard-reset DOM-only state
    allScenes.forEach(s => s.classList.remove('active'));
    document.getElementById('gridBanner').classList.remove('visible');
    document.getElementById('gridCounter').classList.remove('visible');
    document.getElementById('leapCaption').classList.remove('visible');
    document.getElementById('leapClock').classList.remove('alert');
    document.getElementById('leapClock').textContent = '23:59:59';
    document.getElementById('ssdCounter').classList.remove('alert');
    document.getElementById('ssdCounter').textContent = '0 hrs';
    document.getElementById('ssdCaption').classList.remove('alert');
    document.getElementById('ssdCaption').textContent = 'int16 — max value 32,767';
    sceneSsd.classList.remove('flash-red');

    // Reset grid LEDs / bodies
    if (gridCells) {
      gridCells.select('.cell-led').classed('failing', false);
      gridCells.select('.cell-body').attr('stroke', '#30363d').style('opacity', 1);
    }
    // Reset SSD scene
    if (ssdSvg) {
      const singleSel = ssdSvg.select('g.ssd-single');
      singleSel.select('rect.ssd-body')
        .attr('fill', '#161b22').attr('stroke', '#30363d');
      singleSel.select('circle.ssd-led').classed('bricked', false);
      singleSel.select('text.ssd-label').classed('bricked', false);
      singleSel.select('text.ssd-bricked-text').style('opacity', 0);
      ssdSvg.select('rect.ssd-progress').attr('width', 0).attr('fill', '#58a6ff');
    }
    // Reset cities
    if (cityG) cityG.selectAll('circle.city-dot').attr('class', 'city-dot');
    bitEls.forEach(b => {
      b.classList.remove('sign-flip', 'sign-warn', 'on', 'overflow');
      b.textContent = '0';
    });

    clickHint.textContent = 'Tap or press → to advance';
    updateCounter();
  }

  /* ── Debounce ── */
  let lastAdvance = 0;
  const DEBOUNCE_MS = 350;
  function handleAdvance() {
    const now = Date.now();
    if (now - lastAdvance < DEBOUNCE_MS) return;
    lastAdvance = now;
    if (currentStep >= TOTAL_STEPS) reset();
    else advance();
  }

  stage.addEventListener('click', handleAdvance);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      handleAdvance();
    }
    if (e.key === 'ArrowLeft' || e.key === 'r') {
      reset();
    }
  });

  updateCounter();
})();
