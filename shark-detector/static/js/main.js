// When the DOM is fully loaded, attach event listeners
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const resultsDiv = document.getElementById('results');
  
    if (uploadForm) {
      uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
  
        // Create a loader element
        const loader = document.createElement('div');
        loader.classList.add('loader', 'my-4');
        resultsDiv.innerHTML = '';
        resultsDiv.appendChild(loader);
  
        const formData = new FormData(this);
        try {
          const response = await fetch('/upload', {
            method: 'POST',
            body: formData
          });
          const result = await response.json();
  
          // Clear loader and show results
          resultsDiv.innerHTML = '';
          if(result.error) {
            resultsDiv.innerHTML = `<p class="text-red-600">${result.error}</p>`;
          } else {
            for (const [filename, data] of Object.entries(result)) {
              resultsDiv.innerHTML += `<div class="border p-4 mb-4 rounded">
                  <h3 class="font-bold">${filename}</h3>
                  <pre>${JSON.stringify(data, null, 2)}</pre>
              </div>`;
            }
          }
        } catch (error) {
          resultsDiv.innerHTML = `<p class="text-red-600">An error occurred: ${error.message}</p>`;
        }
      });
    }
  });
  