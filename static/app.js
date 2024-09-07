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

function pollTaskStatus(taskID) {
    const url = `/api/task_status/${taskID}`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.state === 'SUCCESS') {
                // When task is successful, update the page with the result
                document.getElementById('acneType').textContent = `Detected Acne Type: ${data.result.acneType}`;
                document.getElementById('acneType').style.display = 'block';
                document.getElementById('recommendations').textContent = `Recommended Products: ${data.result.recommendations.join(', ')}`;
                document.getElementById('recommendations').style.display = 'block';
            } else if (data.state === 'PENDING') {
                // If task is still pending, poll again after 1 second
                setTimeout(() => pollTaskStatus(taskID), 1000);
            }
        })
        .catch(error => console.error('Error polling task status:', error));
}

document.getElementById('imageUploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const imageData = new FormData();
    imageData.append('image', document.getElementById('imageInput').files[0]);

    // Send the image to the server and get the task ID
    fetch('/api/analyze', {
        method: 'POST',
        body: imageData
    })
    .then(response => response.json())
    .then(data => {
        const taskID = data.task_id;
        pollTaskStatus(taskID);  // Start polling for task result
    })
    .catch(error => console.error('Error during image upload:', error));
});
