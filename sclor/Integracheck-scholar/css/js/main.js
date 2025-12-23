function goToUpload() {
  window.location.href = "upload.html";
}

function analyzeDocument() {
  // Simulate processing delay
  document.body.style.opacity = "0.6";

  setTimeout(() => {
    window.location.href = "result.html";
  }, 1200);
}

function analyzeAgain() {
  window.location.href = "upload.html";
}
