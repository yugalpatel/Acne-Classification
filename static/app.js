function previewImage() {
    var preview = document.getElementById('preview');
    var file = document.getElementById('imageInput').files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        preview.src = reader.result;
        preview.style.display = 'block';
    };

    if (file) {
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
        preview.style.display = 'none';
    }
}

document.getElementById('imageUploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const imageData = new FormData();
    imageData.append('image', document.getElementById('imageInput').files[0]);

    fetch('/api/analyze', {
        method: 'POST',
        body: imageData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('acneType').textContent = `Detected Acne Type: ${data.acneType}`;
        document.getElementById('acneType').style.display = 'block';
        document.getElementById('recommendations').textContent = `Recommended Products: ${data.recommendations.join(', ')}`;
        document.getElementById('recommendations').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
});
