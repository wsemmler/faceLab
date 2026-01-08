// ===================== MODEL LOADING =====================
async function loadModels() {
  const modelPaths = [
    faceapi.nets.ssdMobilenetv1,
    faceapi.nets.tinyFaceDetector,
    faceapi.nets.faceLandmark68Net,
    faceapi.nets.ageGenderNet,
    faceapi.nets.faceExpressionNet,
    faceapi.nets.faceRecognitionNet
  ];

  await Promise.all(modelPaths.map(net => net.loadFromUri('./models')));
  console.log('All models loaded');
}
loadModels();

// ===================== HELPER FUNCTIONS =====================
async function readImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.src = reader.result;
      img.onload = () => resolve(img);
      img.onerror = reject;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function detectSingleFaceWithDescriptor(img) {
  try {
    return await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor()
      .withAgeAndGender()
      .withFaceExpressions();
  } catch (err) {
    console.error(err);
    return null;
  }
}

// ===================== FACE MATCH =====================
const referenceInput = document.getElementById('referenceInput');
const testInput = document.getElementById('testInput');
const referenceImage = document.getElementById('referenceImage');
const testImage = document.getElementById('testImage');
const matchButton = document.getElementById('matchButton');
const predictionInput = document.getElementById('prediction');
const distanceTable = document.querySelector('#distanceTable tbody');

let referenceDescriptor = null;

// Helper: Load image as HTMLImageElement
async function readImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.src = reader.result;
      img.onload = () => resolve(img);
      img.onerror = reject;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Helper: Compute face descriptor
async function getFaceDescriptor(file) {
  const img = await readImage(file);
  const detection = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) {
    alert('No face detected!');
    return null;
  }
  return detection.descriptor;
}

// Helper: Cosine similarity
function cosineSimilarity(desc1, desc2) {
  const dot = desc1.reduce((sum, val, i) => sum + val * desc2[i], 0);
  const norm1 = Math.sqrt(desc1.reduce((sum, val) => sum + val * val, 0));
  const norm2 = Math.sqrt(desc2.reduce((sum, val) => sum + val * val, 0));
  return dot / (norm1 * norm2);
}

// Handle reference image upload
referenceInput.addEventListener('change', async (e) => {
  if (!e.target.files.length) return;
  referenceImage.src = URL.createObjectURL(e.target.files[0]);
  referenceDescriptor = await getFaceDescriptor(e.target.files[0]);
});

// Handle test image upload
testInput.addEventListener('change', (e) => {
  if (!e.target.files.length) return;
  testImage.src = URL.createObjectURL(e.target.files[0]);
});

// Run face match with multiple metrics
matchButton.addEventListener('click', async () => {
  if (!referenceDescriptor || !testInput.files.length) {
    alert('Please upload both reference and test images.');
    return;
  }

  const testDescriptor = await getFaceDescriptor(testInput.files[0]);
  if (!testDescriptor) return;

  // Compute metrics
  const euclidean = faceapi.euclideanDistance(referenceDescriptor, testDescriptor);
  const cosine = cosineSimilarity(referenceDescriptor, testDescriptor);
  const confidence = Math.exp(-euclidean * euclidean / 0.08); // transform distance to confidence
  const finalScore = 0.5 * (1 - euclidean) + 0.5 * cosine; // weighted combination

  // Decision
  predictionInput.value = finalScore > 0.7 ? "Same Person" : "Different Person";

  // Update table
  distanceTable.innerHTML = `
    <tr><td>Euclidean Distance</td><td>${euclidean.toFixed(4)}</td></tr>
    <tr><td>Cosine Similarity</td><td>${cosine.toFixed(4)}</td></tr>
    <tr><td>Confidence Score</td><td>${confidence.toFixed(4)}</td></tr>
    <tr><td>Final Score</td><td>${finalScore.toFixed(4)}</td></tr>
  `;
});

// ===================== MULTIPLE FACE LANDMARKS =====================
const multiImageUpload = document.getElementById('imageUpload');
const multiImage = document.getElementById('inputImage');
const multiAnalyzeBtn = document.getElementById('analyzeBtn');
const multiStatusText = document.getElementById('multiStatus');
const multiContainer = document.getElementById('imageContainer');
const detectionScores = document.getElementById('detectionScores');

function clearMultiCanvas() {
  multiContainer.querySelectorAll('canvas').forEach(c => c.remove());
  detectionScores.innerHTML = '';
}

multiImageUpload.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  clearMultiCanvas();
  multiImage.src = URL.createObjectURL(file);
  multiStatusText.innerText = '';
});

async function analyzeMultiFaces() {
  if (!multiImage.src) return;

  if (!multiImage.complete || multiImage.naturalWidth === 0) {
    await new Promise(resolve => multiImage.onload = resolve);
  }

  clearMultiCanvas();

  const detections = await faceapi
    .detectAllFaces(multiImage, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();

  const canvas = faceapi.createCanvasFromMedia(multiImage);
  multiContainer.appendChild(canvas);
  faceapi.matchDimensions(canvas, { width: multiImage.width, height: multiImage.height });
  const resizedDetections = faceapi.resizeResults(detections, { width: multiImage.width, height: multiImage.height });

  faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

  resizedDetections.forEach((det, i) => {
    new faceapi.draw.DrawBox(det.detection.box, { label: `Face ${i + 1}` }).draw(canvas);
    const li = document.createElement('li');
    li.textContent = `Face ${i + 1} Confidence: ${det.detection.score.toFixed(4)}`;
    detectionScores.appendChild(li);
  });

  multiStatusText.innerText = `Detected ${detections.length} face(s)`;
}

multiAnalyzeBtn.addEventListener('click', analyzeMultiFaces);

// ===================== EMOTION & AGE/GENDER DETECTION =====================
const imageInput = document.getElementById('imageInput');
const imageDisplay = document.getElementById('imageDisplay');
const analyzeButton = document.getElementById('analyzeButton');

let faceData = null;

async function analyzeFace(file) {
  const img = await readImage(file);
  imageDisplay.src = URL.createObjectURL(file);

  const detection = await detectSingleFaceWithDescriptor(img);
  if (!detection) {
    faceData = null;
    return alert('No face detected!');
  }

  const expressions = detection.expressions || {};
  const dominantEmotion = Object.keys(expressions).reduce((a, b) =>
    expressions[a] > expressions[b] ? a : b
  , '');

  faceData = {
    age: detection.age.toFixed(0),
    gender: detection.gender,
    emotion: dominantEmotion
  };
}

function displayFaceData() {
  if (!faceData) return;
  document.getElementById('ageValue').textContent = faceData.age;
  document.getElementById('genderValue').textContent = faceData.gender;
  document.getElementById('emotionValue').textContent = faceData.emotion;
}

imageInput.addEventListener('change', e => analyzeFace(e.target.files[0]));
analyzeButton.addEventListener('click', displayFaceData);

// ===================== BLAZEFACE WEBCAM =====================
const blazeVideo = document.getElementById('blazeVideo');
const blazeCanvas = document.getElementById('blazeCanvas');
const blazeCtx = blazeCanvas.getContext('2d');
let blazeModel;
let blazeInterval;
let previousCenters = {}; // Track previous positions for velocity

// Start webcam & load BlazeFace
async function startBlazeFaceStream() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  blazeVideo.srcObject = stream;

  blazeVideo.addEventListener('loadeddata', async () => {
    blazeCanvas.width = blazeVideo.videoWidth;
    blazeCanvas.height = blazeVideo.videoHeight;

    blazeModel = await blazeface.load();
    console.log('BlazeFace loaded');

    if (!blazeInterval) blazeInterval = setInterval(detectBlazeFaces, 100);
  });
}

// Detect faces, draw boxes, and update metrics
async function detectBlazeFaces() {
  if (!blazeModel || blazeVideo.readyState < 2) return;

  const predictions = await blazeModel.estimateFaces(blazeVideo, false);

  // Draw video and boxes
  blazeCtx.clearRect(0, 0, blazeCanvas.width, blazeCanvas.height);
  blazeCtx.drawImage(blazeVideo, 0, 0, blazeCanvas.width, blazeCanvas.height);

  predictions.forEach(pred => {
    const [x1, y1] = pred.topLeft;
    const [x2, y2] = pred.bottomRight;

    blazeCtx.strokeStyle = 'red';
    blazeCtx.lineWidth = 2;
    blazeCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    blazeCtx.fillStyle = 'red';
    pred.landmarks.forEach(l => {
      blazeCtx.fillRect(l[0] - 2, l[1] - 2, 4, 4);
    });
  });

  // Update metrics table
  updateFaceMetricsTable(predictions);
}

// Update the Face Metrics Table
function updateFaceMetricsTable(predictions) {
  const tbody = document.querySelector('#faceMetricsTable tbody');
  tbody.innerHTML = ''; // Clear previous rows

  predictions.forEach((pred, i) => {
    const [x1, y1] = pred.topLeft;
    const [x2, y2] = pred.bottomRight;

    const width = x2 - x1;
    const height = y2 - y1;
    const centerX = x1 + width / 2;
    const centerY = y1 + height / 2;

    // Velocity
    let vx = 0, vy = 0;
    if (previousCenters[i]) {
      vx = centerX - previousCenters[i].x;
      vy = centerY - previousCenters[i].y;
    }
    previousCenters[i] = { x: centerX, y: centerY };

    // Eye distance & tilt
    const leftEye = pred.landmarks[1];
    const rightEye = pred.landmarks[2];
    const eyeDistance = Math.hypot(rightEye[0] - leftEye[0], rightEye[1] - leftEye[1]);
    const tilt = Math.atan2(rightEye[1] - leftEye[1], rightEye[0] - leftEye[0]) * 180 / Math.PI;

    // Add row
    const row = document.createElement('tr');
    row.innerHTML = `
      <td style="padding:4px 8px">${i}</td>
      <td style="padding:4px 8px">${width.toFixed(1)}</td>
      <td style="padding:4px 8px">${height.toFixed(1)}</td>
      <td style="padding:4px 8px">${eyeDistance.toFixed(1)}</td>
      <td style="padding:4px 8px">${tilt.toFixed(1)}</td>
      <td style="padding:4px 8px">${vx.toFixed(1)}</td>
      <td style="padding:4px 8px">${vy.toFixed(1)}</td>
    `;
    tbody.appendChild(row);
  });
}

// Start detection when tab is shown
document.getElementById('blazeface-tab').addEventListener('shown.bs.tab', () => {
  if (!blazeModel) {
    startBlazeFaceStream();
  } else if (!blazeInterval) {
    blazeInterval = setInterval(detectBlazeFaces, 100);
  }
});

// Stop detection when tab hidden
document.getElementById('blazeface-tab').addEventListener('hidden.bs.tab', () => {
  if (blazeInterval) {
    clearInterval(blazeInterval);
    blazeInterval = null;
  }
});

// Resize canvas with window
window.addEventListener('resize', () => {
  if (blazeVideo.videoWidth && blazeVideo.videoHeight) {
    blazeCanvas.width = blazeVideo.videoWidth;
    blazeCanvas.height = blazeVideo.videoHeight;
  }
});

// ===================== FACEMESH WEBCAM =====================
const meshVideo = document.getElementById('meshVideo');
const meshCanvas = document.getElementById('meshCanvas');
const meshCtx = meshCanvas.getContext('2d');

let meshModel;
let meshInterval;

// Start webcam + FaceMesh
async function startFaceMesh() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 720, height: 720 },
    audio: false
  });
  meshVideo.srcObject = stream;

  meshVideo.addEventListener('loadeddata', async () => {
    meshCanvas.width = 720;
    meshCanvas.height = 720;

    meshModel = await faceLandmarksDetection.load(
      faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
      { maxFaces: 3 }
    );

    if (!meshInterval) {
      meshInterval = setInterval(runFaceMesh, 30);
    }
  });
}

// Run FaceMesh
async function runFaceMesh() {
  if (!meshModel || meshVideo.readyState < 2) return;

  const faces = await meshModel.estimateFaces({ input: meshVideo });

  meshCtx.clearRect(0, 0, meshCanvas.width, meshCanvas.height);
  meshCtx.drawImage(meshVideo, 0, 0, 720, 720);

  drawMesh(faces);
}

// Draw FaceMesh
function drawMesh(faces) {
  if (!faces.length) return;

  faces.forEach(face => {
    // Bounding box
    meshCtx.strokeStyle = 'yellow';
    meshCtx.lineWidth = 2;
    meshCtx.strokeRect(
      face.boundingBox.topLeft[0],
      face.boundingBox.topLeft[1],
      face.boundingBox.bottomRight[0] - face.boundingBox.topLeft[0],
      face.boundingBox.bottomRight[1] - face.boundingBox.topLeft[1]
    );

    // Landmarks
    const keypoints = face.scaledMesh;
    meshCtx.fillStyle = 'yellow';

    keypoints.forEach(([x, y]) => {
      meshCtx.beginPath();
      meshCtx.arc(x, y, 1.5, 0, 2 * Math.PI);
      meshCtx.fill();
    });

    // Optional triangulation
    /*
    for (let i = 0; i < TRIANGULATION.length / 3; i++) {
      const points = [
        TRIANGULATION[i * 3],
        TRIANGULATION[i * 3 + 1],
        TRIANGULATION[i * 3 + 2],
      ].map(index => keypoints[index]);

      drawPath(meshCtx, points, true);
    }
    */
  });
}

// Draw triangle paths
function drawPath(ctx, points, closePath) {
  const path = new Path2D();
  path.moveTo(points[0][0], points[0][1]);
  points.slice(1).forEach(p => path.lineTo(p[0], p[1]));
  if (closePath) path.closePath();

  ctx.strokeStyle = 'yellow';
  ctx.lineWidth = 0.5;
  ctx.stroke(path);
}
// Start FaceMesh when tab is shown
document.getElementById('facemesh-tab').addEventListener('shown.bs.tab', () => {
  if (!meshModel) startFaceMesh();
  else if (!meshInterval) meshInterval = setInterval(runFaceMesh, 30);
});

// Stop FaceMesh when tab hidden
document.getElementById('facemesh-tab').addEventListener('hidden.bs.tab', () => {
  if (meshInterval) {
    clearInterval(meshInterval);
    meshInterval = null;
  }
});
