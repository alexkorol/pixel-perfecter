document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('spriteUpload');
    const methodSelect = document.getElementById('segmentationMethod');
    const autoScaleCheckbox = document.getElementById('autoScale');
    const rowsInput = document.getElementById('rows');
    const colsInput = document.getElementById('cols');
    const numColorsInput = document.getElementById('num_colors');
    const manualScaleGroup = document.getElementById('manualScaleGroup');
    const manualScaleInput = document.getElementById('manualScale');
    const blurSizeInput = document.getElementById('blur_size'); // Added
    const generateBtn = document.getElementById('generateBtn');
    const originalPreview = document.getElementById('originalPreview');
    const resultPreview = document.getElementById('resultPreview');
    const statusText = document.getElementById('statusText');
    const downloadLink = document.getElementById('downloadLink');
    const uploadForm = document.getElementById('uploadForm');

    // Preview uploaded image
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                originalPreview.src = event.target.result;
                originalPreview.style.display = 'block';
                resultPreview.style.display = 'none';
                downloadLink.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });

    // Show/hide manual scale input based on checkbox
    autoScaleCheckbox.addEventListener('change', function() {
        manualScaleGroup.style.display = this.checked ? 'none' : 'block';
    });

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            showStatus('Please upload an image first.', 'error');
            return;
        }

        // Basic validation for blur size (must be odd or 0)
        const blurSize = parseInt(blurSizeInput.value, 10);
        if (isNaN(blurSize) || blurSize < 0 || (blurSize > 0 && blurSize % 2 === 0)) {
            showStatus('Blur size must be an odd number >= 1, or 0 to disable.', 'error');
            return;
        }


        showStatus('Processing...');
        generateBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('method', methodSelect.value);
        formData.append('auto_scale', autoScaleCheckbox.checked);
        formData.append('rows', rowsInput.value);
        formData.append('cols', colsInput.value);
        formData.append('num_colors', numColorsInput.value);
        formData.append('blur_size', blurSizeInput.value); // Added

        // Conditionally add manual scale
        if (!autoScaleCheckbox.checked) {
            formData.append('manual_scale', manualScaleInput.value);
        }

        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(text || response.statusText)
                });
            }
            return response.blob();
        })
        .then(imageBlob => {
            const imageUrl = URL.createObjectURL(imageBlob);
            resultPreview.src = imageUrl;
            resultPreview.style.display = 'block';

            downloadLink.href = imageUrl;
            downloadLink.download = 'pixel-perfected-' + file.name;
            downloadLink.style.display = 'inline-block';

            showStatus('Success!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            showStatus(`Error: ${error.message}`, 'error');
            resultPreview.style.display = 'none';
            downloadLink.style.display = 'none';
        })
        .finally(() => {
            generateBtn.disabled = false;
        });
    });

    function showStatus(message, type = '') {
        statusText.textContent = message;
        statusText.className = type;
    }
});