// Gentle 3D sensor-field background (Three.js points wave). Falls back silently if WebGL/CDN unavailable.
(function () {
  try {
    if (typeof THREE === "undefined") return;
    const canvas = document.getElementById("bg3d");
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);
    camera.position.set(0, 6, 11); camera.lookAt(0, 0, 0);
    const N = 46, geo = new THREE.BufferGeometry(), pos = new Float32Array(N * N * 3);
    let k = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { pos[k++] = (i - N / 2) * 0.55; pos[k++] = 0; pos[k++] = (j - N / 2) * 0.55; }
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0x6366f1, size: 0.08, transparent: true, opacity: 0.35 }));
    scene.add(pts);
    function size() { renderer.setSize(innerWidth, innerHeight); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
    size(); addEventListener("resize", size);
    (function anim(t) {
      requestAnimationFrame(anim);
      const a = geo.attributes.position.array;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const idx = (i * N + j) * 3;
        a[idx + 1] = Math.sin(i * 0.35 + t * 0.0011) * 0.45 + Math.cos(j * 0.3 + t * 0.0009) * 0.45;
      }
      geo.attributes.position.needsUpdate = true;
      pts.rotation.y = t * 0.00003;
      renderer.render(scene, camera);
    })(0);
  } catch (e) { /* gradient background in CSS remains */ }
})();
