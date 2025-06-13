document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('recommendation-form');
    const userIdInput = document.getElementById('user_id');
    const errorMessage = document.getElementById('error-message');
    const loadingSpinner = document.getElementById('loading');
    const recommendationsContainer = document.getElementById('recommendations');
    const backgroundOverlay = document.querySelector('.background-overlay');

    // Parallax effect for background
    document.addEventListener('mousemove', (e) => {
        const moveX = (e.clientX / window.innerWidth - 0.5) * 20;
        const moveY = (e.clientY / window.innerHeight - 0.5) * 20;
        backgroundOverlay.style.transform = `translate(${moveX}px, ${moveY}px)`;
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        errorMessage.textContent = '';
        recommendationsContainer.innerHTML = '';
        loadingSpinner.style.display = 'block';

        const userId = userIdInput.value;

        try {
            // Fetch recommendations from Flask
            const response = await fetch('/', {
                method: 'POST',
                body: new FormData(form),
            });
            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error);
            }

            const recommendations = data.recommendations;

            if (recommendations.length === 0) {
                errorMessage.textContent = 'No recommendations found for this user ID.';
                loadingSpinner.style.display = 'none';
                return;
            }

            // Fetch images for each anime
            for (const anime of recommendations) {
                const card = await createAnimeCard(anime);
                recommendationsContainer.appendChild(card);
            }

            loadingSpinner.style.display = 'none';
        } catch (error) {
            errorMessage.textContent = error.message;
            loadingSpinner.style.display = 'none';
        }
    });

    async function createAnimeCard(anime) {
        const card = document.createElement('div');
        card.className = 'anime-card';

        // Fetch image from Jikan API
        let imageUrl = 'https://via.placeholder.com/250x300?text=Image+Not+Found';
        try {
            const response = await fetch(`https://api.jikan.moe/v4/anime?q=${encodeURIComponent(anime)}&limit=1`);
            const data = await response.json();
            if (data.data && data.data.length > 0) {
                imageUrl = data.data[0].images.jpg.large_image_url || imageUrl;
            }
        } catch (error) {
            console.warn(`Failed to fetch image for ${anime}: ${error}`);
        }

        card.innerHTML = `
            <img src="${imageUrl}" alt="${anime}" loading="lazy">
            <div class="anime-card-content">
                <h3>${anime}</h3>
                <p>Synopsis not available in this demo. Watch now to explore!</p>
                <a href="#" class="watch-now">Watch Now</a>
            </div>
        `;
        return card;
    }
});