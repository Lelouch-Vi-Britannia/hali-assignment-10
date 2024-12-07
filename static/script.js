document.getElementById("query_type").addEventListener("change", function() {
    const queryType = this.value;
    const textQueryGroup = document.getElementById("text-query-group");
    const imageQueryGroup = document.getElementById("image-query-group");
    const weightGroup = document.getElementById("weight-group");
    const pcaGroup = document.getElementById("pca-group");
    const pcaKGroup = document.getElementById("pca-k-group");

    if (queryType === "text") {
        // Text only
        textQueryGroup.style.display = "block";
        imageQueryGroup.style.display = "none";
        weightGroup.style.display = "none";
        pcaGroup.style.display = "none";
        pcaKGroup.style.display = "none";
    } else if (queryType === "image") {
        // Image only
        textQueryGroup.style.display = "none";
        imageQueryGroup.style.display = "block";
        weightGroup.style.display = "none";
        pcaGroup.style.display = "block";
        pcaKGroup.style.display = "block";
    } else if (queryType === "hybrid") {
        // Hybrid query
        textQueryGroup.style.display = "block";
        imageQueryGroup.style.display = "block";
        weightGroup.style.display = "block";
        pcaGroup.style.display = "none";
        pcaKGroup.style.display = "none";
    }
});

// No need to listen for "use_pca" change now since we always show pca_k for image queries
// The user can ignore pca_k if not using PCA, the server-side will handle it.

document.getElementById("search-form").addEventListener("submit", async function(event) {
    event.preventDefault();
    const formData = new FormData(this);

    const response = await fetch("/search", {
        method: "POST",
        body: formData
    });

    const resultsDiv = document.getElementById("results");
    const imageResults = document.getElementById("image-results");
    imageResults.innerHTML = "";

    if (!response.ok) {
        const error = await response.json();
        alert("Error: " + error.error);
        return;
    }

    const data = await response.json();
    resultsDiv.style.display = "block";
    data.forEach(item => {
        const div = document.createElement("div");
        div.className = "image-result";
        const img = document.createElement("img");
        img.src = item.image_url;
        const caption = document.createElement("div");
        caption.textContent = `Score: ${item.score.toFixed(4)}`;
        div.appendChild(img);
        div.appendChild(caption);
        imageResults.appendChild(div);
    });
});
