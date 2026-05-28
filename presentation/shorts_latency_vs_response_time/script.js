(() => {
  const stage = document.getElementById('stage');
  const voText = document.getElementById('voText');
  const stepCounter = document.getElementById('stepCounter');
  const clickHint = document.getElementById('clickHint');

  const sceneHook = document.querySelector('.scene-hook');
  const sceneSeq = document.querySelector('.scene-seq');
  const sceneTakeaway = document.querySelector('.scene-takeaway');

  const seqHeader = document.querySelector('.seq-header');
  const llYou = document.querySelector('.ll-you');
  const llOpenai = document.querySelector('.ll-openai');
  const arrowReq = document.querySelector('.arrow-req');
  const arrowRes = document.querySelector('.arrow-res');

  const barNet1 = document.querySelector('.bar-net1');
  const barQueue = document.querySelector('.bar-queue');
  const barProcess = document.querySelector('.bar-process');
  const barNet2 = document.querySelector('.bar-net2');
  const allBars = [barNet1, barQueue, barProcess, barNet2];

  const braceResponse = document.querySelector('.brace-response');
  const braceLatency = document.querySelector('.brace-latency');

  const TOTAL_STEPS = 9;
  let currentStep = 0;

  const voiceovers = [
    '',
    '', // hook — no VO text overlay
    'You ask a question to ChatGPT. It\'s bundled into a network packet and sent over the wire.',
    'It takes time to reach the OpenAI server. That\'s Network Latency.',
    'The server is busy — no GPUs free. Your query waits in a queue. That wait is Queueing Latency.',
    'GPUs become free. Your query is processed. This is Service Time — the real work.',
    'The answer travels back over the network. Another round of Network Latency.',
    'Response Time is everything — from sending the request to getting the answer back.',
    'Latency is just the waiting — network delays + queueing. Not the processing.',
    '',
  ];

  function showScene(el) {
    [sceneHook, sceneSeq, sceneTakeaway].forEach(s => s.classList.remove('active'));
    el.classList.add('active');
  }

  function setVO(step) {
    if (voiceovers[step]) {
      voText.textContent = voiceovers[step];
      voText.classList.add('visible');
    } else {
      voText.classList.remove('visible');
    }
  }

  function updateCounter() {
    stepCounter.textContent = `${currentStep} / ${TOTAL_STEPS}`;
  }

  /* ── Step handlers ── */

  function step1() {
    showScene(sceneHook);
  }

  function step2() {
    showScene(sceneSeq);
    seqHeader.classList.add('visible');
    llYou.classList.add('visible');
    llOpenai.classList.add('visible');
    arrowReq.classList.add('visible');
  }

  function step3() {
    barNet1.classList.add('visible');
  }

  function step4() {
    barQueue.classList.add('visible');
  }

  function step5() {
    barProcess.classList.add('visible');
  }

  function step6() {
    barNet2.classList.add('visible');
    arrowRes.classList.add('visible');
  }

  function step7() {
    // Show Response Time bracket spanning everything
    braceResponse.classList.add('visible');
  }

  function step8() {
    // Dim service time, show Latency bracket
    barProcess.classList.add('dimmed');

    const seqBody = document.querySelector('.seq-body');
    const bodyRect = seqBody.getBoundingClientRect();
    const net1Rect = barNet1.getBoundingClientRect();
    const net2Rect = barNet2.getBoundingClientRect();

    const topPct = ((net1Rect.top - bodyRect.top) / bodyRect.height) * 100;
    const bottomPct = ((bodyRect.bottom - net2Rect.bottom) / bodyRect.height) * 100;
    braceLatency.style.top = topPct + '%';
    braceLatency.style.bottom = bottomPct + '%';

    braceLatency.classList.add('visible');
  }

  function step9() {
    showScene(sceneTakeaway);
  }

  const steps = [null, step1, step2, step3, step4, step5, step6, step7, step8, step9];

  function advance() {
    if (currentStep >= TOTAL_STEPS) return;
    currentStep++;
    steps[currentStep]();
    setVO(currentStep);
    updateCounter();
    if (currentStep >= TOTAL_STEPS) {
      clickHint.textContent = '← Tap to restart';
    }
  }

  function reset() {
    currentStep = 0;
    [sceneHook, sceneSeq, sceneTakeaway].forEach(s => s.classList.remove('active'));
    seqHeader.classList.remove('visible');
    llYou.classList.remove('visible');
    llOpenai.classList.remove('visible');
    arrowReq.classList.remove('visible');
    arrowRes.classList.remove('visible');
    allBars.forEach(b => { b.classList.remove('visible', 'dimmed'); });
    braceResponse.classList.remove('visible');
    braceLatency.classList.remove('visible');
    voText.classList.remove('visible');
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
    if (currentStep >= TOTAL_STEPS) {
      reset();
      advance();
    } else {
      advance();
    }
  }

  /* ── Event listeners ── */
  stage.addEventListener('click', handleAdvance);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault();
      handleAdvance();
    }
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      reset();
    }
  });

  // Init
  reset();
  stepCounter.textContent = `0 / ${TOTAL_STEPS}`;
})();
