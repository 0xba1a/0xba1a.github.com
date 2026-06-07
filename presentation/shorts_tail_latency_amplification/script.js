(() => {
  // ─────────────────────────────────────────
  // State
  // ─────────────────────────────────────────
  const TOTAL_STEPS = 8;
  let currentStep = 0;
  let lastClick = 0;
  const CLICK_DEBOUNCE = 320;

  const stage = document.getElementById('stage');
  const stepCounter = document.getElementById('stepCounter');
  const clickHint = document.getElementById('clickHint');

  const scenes = {
    hook:      document.querySelector('.scene-hook'),
    complaint: document.querySelector('.scene-complaint'),
    fanout:    document.querySelector('.scene-fanout'),
    grid:      document.querySelector('.scene-grid'),
    amp:       document.querySelector('.scene-amp'),
    curve:     document.querySelector('.scene-curve'),
    takeaway:  document.querySelector('.scene-takeaway'),
  };

  function setActiveScene(name) {
    Object.entries(scenes).forEach(([k, el]) => {
      el.classList.toggle('active', k === name);
    });
  }

  // ─────────────────────────────────────────
  // Scene 3 + 4 — Fan-out + bars
  // ─────────────────────────────────────────
  const FW = 600, FH = 760;
  const fanoutCanvas = d3.select('#fanoutCanvas');
  const fanoutBars   = d3.select('#fanoutBars');
  const fanoutCaption = document.getElementById('fanoutCaption');

  let fanoutSvg = null;
  let barsSvg = null;
  let fanoutBuilt = false;

  // Layout for fan-out diagram
  const USER     = { x: 80,  y: FH / 2, w: 120, h: 120 };
  const BACKEND  = { x: 290, y: FH / 2, w: 170, h: 140 };
  const DB_X     = 500;
  const DB_R     = 32;

  function dbY(i) {
    // 10 DB nodes spread vertically
    const top = 50, bot = FH - 50;
    return top + (bot - top) * (i / 9);
  }

  function buildFanout() {
    if (fanoutBuilt) return;
    fanoutBuilt = true;

    fanoutCanvas.selectAll('*').remove();
    fanoutSvg = fanoutCanvas.append('svg')
      .attr('viewBox', `0 0 ${FW} ${FH}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // Arrowhead defs
    const defs = fanoutSvg.append('defs');
    defs.append('marker')
      .attr('id', 'fo-head')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto-start-reverse')
      .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('class', 'fo-arrowhead');

    defs.append('marker')
      .attr('id', 'fo-head-fan')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8).attr('refY', 5)
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('orient', 'auto-start-reverse')
      .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('class', 'fo-arrowhead fan');

    // ── User node
    const userG = fanoutSvg.append('g').attr('class', 'fo-node user-node');
    userG.append('rect')
      .attr('class', 'fo-node-rect user')
      .attr('x', USER.x - USER.w / 2).attr('y', USER.y - USER.h / 2)
      .attr('width', USER.w).attr('height', USER.h)
      .attr('rx', 16);
    userG.append('text')
      .attr('class', 'fo-node-icon')
      .attr('x', USER.x).attr('y', USER.y - 14)
      .text('👤');
    userG.append('text')
      .attr('class', 'fo-node-label')
      .attr('x', USER.x).attr('y', USER.y + 36)
      .style('font-size', '24px')
      .text('User');

    // ── Backend node
    const beG = fanoutSvg.append('g').attr('class', 'fo-node backend-node');
    beG.append('rect')
      .attr('class', 'fo-node-rect backend')
      .attr('x', BACKEND.x - BACKEND.w / 2).attr('y', BACKEND.y - BACKEND.h / 2)
      .attr('width', BACKEND.w).attr('height', BACKEND.h)
      .attr('rx', 14);
    beG.append('text')
      .attr('class', 'fo-node-label')
      .attr('x', BACKEND.x).attr('y', BACKEND.y - 14)
      .text('Backend');
    beG.append('text')
      .attr('class', 'fo-node-label')
      .attr('x', BACKEND.x).attr('y', BACKEND.y + 26)
      .style('font-size', '22px')
      .style('fill', '#8b949e')
      .text('search');

    // ── 10 DB nodes
    for (let i = 0; i < 10; i++) {
      const y = dbY(i);
      const g = fanoutSvg.append('g').attr('class', `fo-node db-node db-${i}`);
      g.append('circle')
        .attr('class', 'fo-node-rect db')
        .attr('cx', DB_X).attr('cy', y).attr('r', DB_R);
      g.append('text')
        .attr('class', 'fo-node-label')
        .attr('x', DB_X).attr('y', y)
        .style('font-size', '22px')
        .style('font-family', 'JetBrains Mono, ui-monospace, monospace')
        .style('fill', '#3fb950')
        .text(`Q${i + 1}`);
    }

    // ── Static "DB" cluster label
    fanoutSvg.append('text')
      .attr('class', 'fo-node-label')
      .attr('x', DB_X).attr('y', 22)
      .style('font-size', '26px')
      .style('fill', '#3fb950')
      .text('Database');

    // ── User → Backend request arrow (initially hidden)
    fanoutSvg.append('path')
      .attr('class', 'fo-arrow request-arrow')
      .attr('d', `M ${USER.x + USER.w / 2} ${USER.y} L ${BACKEND.x - BACKEND.w / 2} ${BACKEND.y}`)
      .attr('marker-end', 'url(#fo-head)');

    // ── 10 fan-out arrows from Backend to each DB
    for (let i = 0; i < 10; i++) {
      const y = dbY(i);
      fanoutSvg.append('path')
        .attr('class', `fo-arrow fan fanout-arrow fan-${i}`)
        .attr('d', `M ${BACKEND.x + BACKEND.w / 2} ${BACKEND.y} L ${DB_X - DB_R} ${y}`)
        .attr('marker-end', 'url(#fo-head-fan)');
    }
  }

  function resetFanoutAnimations() {
    if (!fanoutSvg) return;
    // Hide request arrow & fan arrows by setting full dash offset
    const req = fanoutSvg.select('.request-arrow').node();
    if (req) {
      const len = req.getTotalLength();
      d3.select(req)
        .attr('stroke-dasharray', `${len} ${len}`)
        .attr('stroke-dashoffset', len)
        .attr('opacity', 1);
    }
    fanoutSvg.selectAll('.fanout-arrow').each(function() {
      const len = this.getTotalLength();
      d3.select(this)
        .attr('stroke-dasharray', `${len} ${len}`)
        .attr('stroke-dashoffset', len)
        .attr('opacity', 1);
    });
  }

  function playFanoutScene() {
    buildFanout();
    resetFanoutAnimations();

    fanoutBars.classed('visible', false);
    fanoutCaption.classList.remove('visible');

    const req = fanoutSvg.select('.request-arrow');
    const reqLen = req.node().getTotalLength();

    req.transition().duration(700).ease(d3.easeCubicOut)
      .attr('stroke-dashoffset', 0)
      .on('end', () => {
        // After request arrives at backend, fan out the 10 arrows
        fanoutSvg.selectAll('.fanout-arrow').each(function(_, i) {
          const node = this;
          const len = node.getTotalLength();
          d3.select(node)
            .transition()
              .delay(80 + i * 70)
              .duration(550)
              .ease(d3.easeCubicOut)
              .attr('stroke-dashoffset', 0);
        });

        setTimeout(() => {
          fanoutCaption.textContent = 'all 10 must finish before the user gets a reply';
          fanoutCaption.classList.add('visible');
        }, 80 + 10 * 70 + 400);
      });
  }

  // ── Scene 4: Bars
  const BW = 600, BH = 760;

  function buildBars() {
    fanoutBars.selectAll('*').remove();
    barsSvg = fanoutBars.append('svg')
      .attr('viewBox', `0 0 ${BW} ${BH}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const padL = 95, padR = 110, padT = 50, padB = 50;
    const innerW = BW - padL - padR;
    const innerH = BH - padT - padB;

    const rowH = innerH / 10;
    const barH = rowH * 0.6;
    const barY0 = padT + (rowH - barH) / 2;

    // Choose Q7 (index 6) as the slow one
    const slowIdx = 6;
    const fastEnd = padL + innerW * 0.34;     // where fast bars end
    const slowEnd = padL + innerW * 0.92;     // where the slow bar ends

    // Wait gate (vertical line at the right where the user is gated)
    const gateX = padL + innerW;

    barsSvg.append('line')
      .attr('class', 'gate-line')
      .attr('x1', gateX).attr('x2', gateX)
      .attr('y1', padT - 10).attr('y2', BH - padB + 10);

    barsSvg.append('text')
      .attr('class', 'gate-label')
      .attr('x', gateX).attr('y', padT - 22)
      .text('user waits ↧');

    for (let i = 0; i < 10; i++) {
      const y = padT + i * rowH + (rowH - barH) / 2;
      const isSlow = i === slowIdx;

      // Label "Q1"…
      barsSvg.append('text')
        .attr('class', `bar-label ${isSlow ? 'slow' : ''}`)
        .attr('x', padL - 18)
        .attr('y', y + barH / 2)
        .attr('text-anchor', 'end')
        .text(`Q${i + 1}`);

      // Track
      barsSvg.append('rect')
        .attr('class', 'bar-track')
        .attr('x', padL).attr('y', y)
        .attr('width', innerW).attr('height', barH);

      // Fill
      barsSvg.append('rect')
        .attr('class', `bar-fill ${isSlow ? 'slow' : 'fast'} bar-fill-${i}`)
        .attr('x', padL).attr('y', y)
        .attr('width', 0).attr('height', barH);

      if (isSlow) {
        barsSvg.append('text')
          .attr('class', 'bar-tag')
          .attr('x', padL + innerW + 14)
          .attr('y', y + barH / 2)
          .attr('opacity', 0)
          .text('p99 hit ⚠');
      }
    }

    return { padL, padR, padT, padB, innerW, innerH, rowH, barH, slowIdx, fastEnd, slowEnd, gateX };
  }

  function playBarsScene() {
    fanoutBars.classed('visible', true);
    const m = buildBars();

    // Animate fast bars
    for (let i = 0; i < 10; i++) {
      if (i === m.slowIdx) continue;
      barsSvg.select(`.bar-fill-${i}`)
        .transition()
          .delay(60 + i * 30)
          .duration(450)
          .ease(d3.easeCubicOut)
          .attr('width', m.fastEnd - m.padL);
    }

    // Animate the slow bar — slower, longer, and red
    barsSvg.select(`.bar-fill-${m.slowIdx}`)
      .transition()
        .delay(120)
        .duration(2200)
        .ease(d3.easeCubicInOut)
        .attr('width', m.slowEnd - m.padL);

    // Reveal "p99 hit ⚠" tag mid-way
    barsSvg.select('.bar-tag')
      .transition()
        .delay(1400)
        .duration(400)
        .attr('opacity', 1);

    // Caption swaps mid-animation
    fanoutCaption.classList.remove('visible');
    setTimeout(() => {
      fanoutCaption.textContent = 'one slow query → the whole user request waits';
      fanoutCaption.classList.add('visible');
    }, 1500);
  }

  // ─────────────────────────────────────────
  // Scene 5 — Math grid (10 × 10 dots)
  // ─────────────────────────────────────────
  const GW = 600, GH = 600;
  const gridCanvas = d3.select('#gridCanvas');
  let gridSvg = null;
  let gridTimers = [];

  function clearGridTimers() {
    gridTimers.forEach(t => clearTimeout(t));
    gridTimers = [];
  }

  function buildGrid() {
    clearGridTimers();
    gridCanvas.selectAll('*').remove();

    gridSvg = gridCanvas.append('svg')
      .attr('viewBox', `0 0 ${GW} ${GH}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const padL = 95, padR = 40, padT = 30, padB = 30;
    const innerW = GW - padL - padR;
    const innerH = GH - padT - padB;
    const cell = Math.min(innerW / 10, innerH / 10);

    const offsetX = padL + (innerW - cell * 10) / 2;
    const offsetY = padT + (innerH - cell * 10) / 2;

    // Row glow rectangles (one per row) + row labels
    for (let r = 0; r < 10; r++) {
      gridSvg.append('rect')
        .attr('class', `row-glow row-glow-${r}`)
        .attr('x', offsetX - 8)
        .attr('y', offsetY + r * cell + cell * 0.08)
        .attr('width', cell * 10 + 16)
        .attr('height', cell * 0.84);

      gridSvg.append('text')
        .attr('class', 'row-label')
        .attr('x', offsetX - 20)
        .attr('y', offsetY + r * cell + cell / 2)
        .text(`U${r + 1}`);
    }

    // 100 dots
    const dotR = cell * 0.36;
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        gridSvg.append('circle')
          .attr('class', `dot dot-${r}-${c}`)
          .attr('cx', offsetX + c * cell + cell / 2)
          .attr('cy', offsetY + r * cell + cell / 2)
          .attr('r', 0)
          .attr('fill', '#3fb950')
          .attr('opacity', 0.85);
      }
    }
    return { dotR };
  }

  function playGridScene() {
    document.getElementById('gridQueryPct').textContent = '0%';
    document.getElementById('gridUserPct').textContent  = '0%';

    const { dotR } = buildGrid();

    // Phase 1: all 100 dots fade in green (~950ms total)
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        gridSvg.select(`.dot-${r}-${c}`)
          .transition()
            .delay(50 + (r * 10 + c) * 6)
            .duration(280)
            .attr('r', dotR);
      }
    }
    const FADE_IN_TOTAL = 50 + 100 * 6 + 280; // ≈ 930ms

    // Phase 2: one random dot turns red — represents the 1% slow query
    const targetRow = Math.floor(Math.random() * 10);
    const targetCol = Math.floor(Math.random() * 10);

    // Step 2a — flip dot red, increment "slow queries" to 1%
    gridTimers.push(setTimeout(() => {
      gridSvg.select(`.dot-${targetRow}-${targetCol}`)
        .transition().duration(320)
        .attr('fill', '#f85149')
        .attr('opacity', 1);
      document.getElementById('gridQueryPct').textContent = '1%';
    }, FADE_IN_TOTAL + 400));

    // Step 2b — light up that user's row, jump "slow users" to 10%
    gridTimers.push(setTimeout(() => {
      gridSvg.select(`.row-glow-${targetRow}`).classed('on', true);
      document.getElementById('gridUserPct').textContent = '10%';
    }, FADE_IN_TOTAL + 1100));
  }

  // ─────────────────────────────────────────
  // Scene 7 — Amplification curve
  // ─────────────────────────────────────────
  const CW = 600, CH = 880;
  const curveCanvas = d3.select('#curveCanvas');
  let curveBuilt = false;

  function buildCurve() {
    if (curveBuilt) return;
    curveBuilt = true;

    const margin = { top: 40, right: 50, bottom: 95, left: 115 };
    const iw = CW - margin.left - margin.right;
    const ih = CH - margin.top - margin.bottom;

    const svg = curveCanvas.append('svg')
      .attr('viewBox', `0 0 ${CW} ${CH}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear().domain([1, 100]).range([0, iw]);
    const yScale = d3.scaleLinear().domain([0, 100]).range([ih, 0]);

    // Axes
    const xAxis = d3.axisBottom(xScale)
      .tickValues([1, 10, 25, 50, 75, 100])
      .tickFormat(d => `${d}`);
    const yAxis = d3.axisLeft(yScale)
      .tickValues([0, 25, 50, 75, 100])
      .tickFormat(d => `${d}%`);

    g.append('g')
      .attr('class', 'curve-axis')
      .attr('transform', `translate(0, ${ih})`)
      .call(xAxis);
    g.append('g')
      .attr('class', 'curve-axis')
      .call(yAxis);

    // Axis titles
    g.append('text')
      .attr('class', 'curve-axis-title')
      .attr('x', iw / 2).attr('y', ih + 75)
      .attr('text-anchor', 'middle')
      .text('fan-out (queries / request)');
    g.append('text')
      .attr('class', 'curve-axis-title')
      .attr('transform', `translate(-82, ${ih / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .text('% of slow users');

    // Curve: 1 - 0.99^n
    const points = [];
    for (let n = 1; n <= 100; n++) {
      points.push({ n, p: (1 - Math.pow(0.99, n)) * 100 });
    }
    const line = d3.line()
      .x(d => xScale(d.n))
      .y(d => yScale(d.p))
      .curve(d3.curveMonotoneX);

    const path = g.append('path')
      .datum(points)
      .attr('class', 'amp-curve')
      .attr('d', line);

    const totalLen = path.node().getTotalLength();
    path
      .attr('stroke-dasharray', `${totalLen} ${totalLen}`)
      .attr('stroke-dashoffset', totalLen);

    // Markers (drawn but hidden, revealed in playCurve)
    const markers = [
      { n: 1,   p: 1,    label: '1 → 1%' },
      { n: 10,  p: 9.6,  label: '10 → ~10%' },
      { n: 100, p: 63.4, label: '100 → ~63%' },
    ];

    markers.forEach((m, i) => {
      const mg = g.append('g')
        .attr('class', `amp-marker-group amp-marker-${i}`)
        .attr('opacity', 0);

      mg.append('circle')
        .attr('class', 'amp-marker')
        .attr('cx', xScale(m.n))
        .attr('cy', yScale(m.p))
        .attr('r', 10);

      const labelX = xScale(m.n) + (m.n === 100 ? -16 : 20);
      const labelY = yScale(m.p) + (m.n === 100 ? -18 : (m.n === 1 ? -18 : -14));
      const anchor = m.n === 100 ? 'end' : 'start';

      mg.append('text')
        .attr('class', 'amp-marker-label')
        .attr('x', labelX).attr('y', labelY)
        .attr('text-anchor', anchor)
        .text(m.label);
    });

    // Save references on the canvas DOM for replay
    curveCanvas.node().__curve = { path, totalLen, g };
  }

  function playCurve() {
    buildCurve();
    const { path, totalLen, g } = curveCanvas.node().__curve;
    // Reset and redraw
    path
      .attr('stroke-dashoffset', totalLen)
      .transition().duration(1400).ease(d3.easeCubicOut)
      .attr('stroke-dashoffset', 0);

    g.selectAll('.amp-marker-group').attr('opacity', 0);
    [0, 1, 2].forEach(i => {
      g.select(`.amp-marker-${i}`)
        .transition().delay(700 + i * 350).duration(400)
        .attr('opacity', 1);
    });
  }

  // ─────────────────────────────────────────
  // Step handlers
  // ─────────────────────────────────────────
  const stepHandlers = {
    1: () => setActiveScene('hook'),
    2: () => setActiveScene('complaint'),
    3: () => {
      setActiveScene('fanout');
      playFanoutScene();
    },
    4: () => {
      setActiveScene('fanout');
      // Make sure the fan-out diagram is visible (in case we jumped)
      buildFanout();
      // Force-complete the request + fan arrows so bars scene reads the full diagram
      fanoutSvg.select('.request-arrow').interrupt().attr('stroke-dashoffset', 0);
      fanoutSvg.selectAll('.fanout-arrow').interrupt().attr('stroke-dashoffset', 0);
      playBarsScene();
    },
    5: () => {
      setActiveScene('grid');
      playGridScene();
    },
    6: () => setActiveScene('amp'),
    7: () => {
      setActiveScene('curve');
      playCurve();
    },
    8: () => setActiveScene('takeaway'),
  };

  // ─────────────────────────────────────────
  // Navigation
  // ─────────────────────────────────────────
  function goToStep(n) {
    if (n < 1 || n > TOTAL_STEPS) return;
    currentStep = n;
    stepCounter.textContent = `${currentStep} / ${TOTAL_STEPS}`;
    clickHint.textContent = currentStep === TOTAL_STEPS
      ? '← Tap to restart'
      : 'Tap or press → to advance';
    stepHandlers[n]();
  }

  function reset() {
    clearGridTimers();
    if (fanoutSvg) {
      fanoutSvg.selectAll('.request-arrow, .fanout-arrow').interrupt();
    }
    fanoutBars.classed('visible', false);
    fanoutCaption.classList.remove('visible');
    goToStep(1);
  }

  function advance() {
    const now = Date.now();
    if (now - lastClick < CLICK_DEBOUNCE) return;
    lastClick = now;
    if (currentStep >= TOTAL_STEPS) reset();
    else goToStep(currentStep + 1);
  }

  // ─────────────────────────────────────────
  // Events
  // ─────────────────────────────────────────
  stage.addEventListener('click', advance);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      advance();
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      if (currentStep > 1) goToStep(currentStep - 1);
    } else if (e.key === 'r' || e.key === 'R') {
      reset();
    }
  });

  // ─────────────────────────────────────────
  // Init
  // ─────────────────────────────────────────
  goToStep(1);
})();
