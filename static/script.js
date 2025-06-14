class AnimeRecommendationSystem {
    constructor() {
        console.log('Initializing AnimeRecommendationSystem');
        this.form = document.getElementById('recommendation-form');
        this.loadingSection = document.getElementById('loading-section');
        this.errorMessage = document.getElementById('error-message');
        this.recommendationsSection = document.getElementById('recommendations-section');
        this.recommendationsGrid = document.getElementById('recommendations-grid');
        this.imageCache = new Map();
        this.metadataCache = new Map();
        this.apiRequestQueue = [];
        this.isProcessingQueue = false;
        this.progressBar = document.getElementById('progress-bar');
        this.inputTooltip = document.getElementById('input-tooltip');
        
        if (!this.form || !this.recommendationsGrid) {
            console.error('Required DOM elements not found. Check HTML IDs.');
        }
        
        this.init();
    }

    init() {
        console.log('Setting up system');
        this.createFloatingElements();
        this.setupEventListeners();
        this.setupParallaxEffect();
        this.setupProgressBar();
        this.setupBackToTop();
    }

    createFloatingElements() {
        const container = document.getElementById('floating-elements');
        if (!container) return;
        const elementCount = 30;
        
        for (let i = 0; i < elementCount; i++) {
            const element = document.createElement('div');
            element.className = 'floating-element';
            element.style.left = Math.random() * 100 + '%';
            element.style.animationDelay = Math.random() * 25 + 's';
            element.style.animationDuration = (Math.random() * 15 + 15) + 's';
            container.appendChild(element);
        }
    }

    setupEventListeners() {
        if (!this.form) return;
        this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        const userIdInput = document.getElementById('user_id');
        const clearButton = document.querySelector('.clear-button');
        if (userIdInput) {
            userIdInput.addEventListener('input', this.validateInput.bind(this));
        }
        if (clearButton) {
            clearButton.addEventListener('click', () => {
                if (userIdInput) {
                    userIdInput.value = '';
                    this.validateInput({ target: userIdInput });
                }
            });
        }
    }

    setupParallaxEffect() {
        let ticking = false;
        
        document.addEventListener('mousemove', (e) => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    const layers = document.querySelectorAll('.background-layer');
                    const mouseX = e.clientX / window.innerWidth;
                    const mouseY = e.clientY / window.innerHeight;
                    
                    layers.forEach((layer, index) => {
                        const speed = (index + 1) * 0.5;
                        const x = (mouseX - 0.5) * speed;
                        const y = (mouseY - 0.5) * speed;
                        layer.style.transform = `translate(${x}px, ${y}px)`;
                    });
                    
                    ticking = false;
                });
                ticking = true;
            }
        });
    }

    setupProgressBar() {
        this.progressInterval = null;
    }

    setupBackToTop() {
        const backToTop = document.getElementById('back-to-top');
        if (!backToTop) return;
        window.addEventListener('scroll', () => {
            backToTop.style.display = window.scrollY > 300 ? 'block' : 'none';
        });
        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    validateInput(e) {
        const value = e.target.value;
        const isValid = value && parseInt(value) > 0;
        
        e.target.classList.toggle('invalid', !isValid && value !== '');
        if (this.inputTooltip) {
            this.inputTooltip.textContent = isValid ? '' : 'Please enter a valid positive User ID';
            this.inputTooltip.style.display = isValid ? 'none' : 'block';
        }
        this.hideError();
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        console.log('Form submitted');
        
        const userId = document.getElementById('user_id').value;
        if (!userId || parseInt(userId) <= 0) {
            this.showError('Please enter a valid User ID');
            return;
        }

        this.showLoading();
        this.startProgressBar();
        this.hideError();
        this.hideRecommendations();

        try {
            const recommendations = await this.fetchRecommendations(userId);
            console.log('Recommendations received:', recommendations);
            
            if (!recommendations || recommendations.length === 0) {
                throw new Error('No recommendations found for this user ID');
            }

            await this.displayRecommendations(recommendations);
        } catch (error) {
            this.showError(error.message);
            console.error('Recommendation error:', error);
        } finally {
            this.hideLoading();
            this.stopProgressBar();
        }
    }

    startProgressBar() {
        if (!this.progressBar) return;
        let progress = 0;
        this.progressBar.style.width = '0%';
        this.progressInterval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 100) progress = 95;
            this.progressBar.style.width = `${progress}%`;
        }, 300);
    }

    stopProgressBar() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            if (this.progressBar) {
                this.progressBar.style.width = '100%';
                setTimeout(() => {
                    this.progressBar.style.width = '0%';
                }, 500);
            }
        }
    }

    async fetchRecommendations(userId) {
        console.log('Fetching recommendations for user ID:', userId);
        const formData = new FormData();
        formData.append('user_id', userId);

        const response = await fetch('/', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        console.log('Server response:', data);

        if (!data.success) {
            throw new Error(data.error || 'Failed to get recommendations');
        }

        return data.recommendations;
    }

    async fetchAnimeDetails(animeName, retries = 3) {
        if (this.metadataCache.has(animeName)) {
            return this.metadataCache.get(animeName);
        }

        for (let attempt = 0; attempt < retries; attempt++) {
            try {
                const response = await fetch(`https://api.jikan.moe/v4/anime?q=${encodeURIComponent(animeName)}&limit=1`);
                if (!response.ok) throw new Error('Jikan API failed');
                
                const data = await response.json();
                if (data.data && data.data.length > 0) {
                    const anime = data.data[0];
                    const details = {
                        synopsis: anime.synopsis || 'No synopsis available',
                        genres: anime.genres?.map(g => g.name) || [],
                        rating: anime.score || 'N/A',
                        image_url: anime.images.jpg.large_image_url || anime.images.jpg.image_url
                    };
                    this.metadataCache.set(animeName, details);
                    return details;
                }
            } catch (error) {
                console.warn(`Attempt ${attempt + 1} failed for ${animeName}:`, error);
                if (attempt < retries - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
                }
            }
        }

        const fallback = {
            synopsis: 'No synopsis available',
            genres: [],
            rating: 'N/A',
            image_url: this.generateFallbackImage(animeName)
        };
        this.metadataCache.set(animeName, fallback);
        return fallback;
    }

    async fetchAnimeImage(anime, retries = 3) {
        const animeName = typeof anime === 'string' ? anime : anime.title || anime;
        
        if (this.imageCache.has(animeName)) {
            return this.imageCache.get(animeName);
        }

        return new Promise((resolve) => {
            this.apiRequestQueue.push(async () => {
                try {
                    const details = await this.fetchAnimeDetails(animeName);
                    const imageUrl = details.image_url;
                    this.imageCache.set(animeName, imageUrl);
                    resolve(imageUrl);
                } catch (error) {
                    console.warn(`Failed to fetch image for ${animeName}:`, error);
                    const fallbackUrl = this.generateFallbackImage(animeName);
                    this.imageCache.set(animeName, fallbackUrl);
                    resolve(fallbackUrl);
                }
            });

            if (!this.isProcessingQueue) {
                this.processImageQueue();
            }
        });
    }

    async processImageQueue() {
        this.isProcessingQueue = true;
        
        while (this.apiRequestQueue.length > 0) {
            const request = this.apiRequestQueue.shift();
            await request();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        this.isProcessingQueue = false;
    }

    async searchAnimeImage(animeName, retries) {
        const apis = [
            () => this.fetchFromJikan(animeName),
            () => this.fetchFromAniList(animeName),
            () => this.fetchFromKitsu(animeName)
        ];

        for (const apiCall of apis) {
            for (let attempt = 0; attempt < retries; attempt++) {
                try {
                    const result = await apiCall();
                    if (result) return result;
                } catch (error) {
                    console.warn(`API attempt ${attempt + 1} failed:`, error);
                    if (attempt < retries - 1) {
                        await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
                    }
                }
            }
        }

        throw new Error(`Failed to fetch image for ${animeName}`);
    }

    async fetchFromJikan(animeName) {
        const response = await fetch(`https://api.jikan.moe/v4/anime?q=${encodeURIComponent(animeName)}&limit=1`);
        if (!response.ok) throw new Error('Jikan API failed');
        
        const data = await response.json();
        if (data.data && data.data.length > 0) {
            return data.data[0].images.jpg.large_image_url || data.data[0].images.jpg.image_url;
        }
        return null;
    }

    async fetchFromAniList(animeName) {
        const query = `
            query ($search: String) {
                Media (search: $search, type: ANIME) {
                    coverImage {
                        large
                        medium
                    }
                }
            }
        `;

        const response = await fetch('https://graphql.anilist.co', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                variables: { search: animeName }
            })
        });

        if (!response.ok) throw new Error('AniList API failed');
        
        const data = await response.json();
        if (data.data && data.data.Media) {
            return data.data.Media.coverImage.large || data.data.Media.coverImage.medium;
        }
        return null;
    }

    async fetchFromKitsu(animeName) {
        const response = await fetch(`https://kitsu.io/api/edge/anime?filter[text]=${encodeURIComponent(animeName)}&page[limit]=1`);
        if (!response.ok) throw new Error('Kitsu API failed');
        
        const data = await response.json();
        if (data.data && data.data.length > 0) {
            return data.data[0].attributes.posterImage?.large || data.data[0].attributes.posterImage?.medium;
        }
        return null;
    }

    generateFallbackImage(animeName) {
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');

        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(1, '#764ba2');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 300, 400);

        ctx.fillStyle = 'white';
        ctx.font = 'bold 20px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        const words = animeName.split(' ');
        let lines = [];
        let currentLine = '';
        
        words.forEach(word => {
            const testLine = currentLine + (currentLine ? ' ' : '') + word;
            if (ctx.measureText(testLine).width > 260) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        });
        lines.push(currentLine);

        const lineHeight = 30;
        const startY = 200 - (lines.length - 1) * lineHeight / 2;
        
        lines.forEach((line, index) => {
            ctx.fillText(line, 150, startY + index * lineHeight);
        });

        return canvas.toDataURL();
    }

    async displayRecommendations(recommendations) {
        if (!this.recommendationsGrid) return;
        this.recommendationsGrid.innerHTML = '';
        console.log('Displaying recommendations:', recommendations);
        
        const cards = await Promise.all(
            recommendations.map(async (anime, index) => {
                const card = await this.createAnimeCard(anime, index);
                return card;
            })
        );

        cards.forEach((card, index) => {
            setTimeout(() => {
                this.recommendationsGrid.appendChild(card);
                requestAnimationFrame(() => {
                    card.classList.add('animate-in');
                });
            }, index * 100);
        });

        this.showRecommendations();
    }

    async createAnimeCard(anime, index) {
        console.log(`Creating card for anime: ${anime}, index: ${index}`);
        const card = document.createElement('div');
        card.className = 'anime-card';
        card.style.animationDelay = `${index * 0.1}s`;

        const details = await this.fetchAnimeDetails(anime);
        const imageUrl = await this.fetchAnimeImage(anime);
        
        card.innerHTML = `
            <div class="card-image-container">
                <img src="${imageUrl}" alt="${anime}" class="card-image" loading="lazy" onerror="this.src='${this.generateFallbackImage(anime)}'">
                <div class="image-overlay"></div>
                <div class="card-actions">
                    <a href="https://www.google.com/search?q=${encodeURIComponent(anime + ' site:crunchyroll.com')}" class="watch-button" target="_blank" title="Search on Google for Crunchyroll page">
                        <i class="fas fa-play"></i> Watch Now
                    </a>
                    <a href="https://myanimelist.net/anime.php?q=${encodeURIComponent(anime)}" class="info-button" target="_blank" title="View on MyAnimeList">
                        <i class="fas fa-info-circle"></i>
                    </a>
                    <button class="info-button favorite-btn" title="Add to favorites">
                        <i class="fas fa-heart"></i>
                    </button>
                </div>
            </div>
            <div class="card-content">
                <h3 class="card-title">${anime}</h3>
                <div class="card-meta">
                    <span class="recommendation-rank">#${index + 1}</span>
                    <span class="rating"><i class="fas fa-star"></i> ${details.rating}</span>
                </div>
                <p class="card-genres">${details.genres.join(', ') || 'No genres available'}</p>
                <p class="card-synopsis ${details.synopsis === 'No synopsis available' ? 'empty' : ''}">${details.synopsis}</p>
            </div>
        `;

        const favoriteBtn = card.querySelector('.favorite-btn');
        favoriteBtn.addEventListener('click', () => {
            console.log(`Favorite button clicked for ${anime}`);
            favoriteBtn.classList.toggle('active');
        });

        const watchButton = card.querySelector('.watch-button');
        if (watchButton) {
            console.log(`Watch Now button created for ${anime} with href: ${watchButton.href}`);
        }

        return card;
    }

    showLoading() {
        if (this.loadingSection) {
            this.loadingSection.style.display = 'flex';
            this.loadingSection.classList.add('fade-in');
        }
    }

    hideLoading() {
        if (this.loadingSection) {
            this.loadingSection.classList.remove('fade-in');
            setTimeout(() => {
                this.loadingSection.style.display = 'none';
            }, 300);
        }
    }

    showError(message) {
        if (this.errorMessage) {
            this.errorMessage.textContent = message;
            this.errorMessage.style.display = 'block';
            this.errorMessage.classList.add('fade-in');
            
            setTimeout(() => this.hideError(), 5000);
        }
    }

    hideError() {
        if (this.errorMessage) {
            this.errorMessage.classList.remove('fade-in');
            setTimeout(() => {
                this.errorMessage.style.display = 'none';
            }, 300);
        }
    }

    showRecommendations() {
        if (this.recommendationsSection) {
            this.recommendationsSection.style.display = 'block';
            this.recommendationsSection.classList.add('fade-in');
        }
    }

    hideRecommendations() {
        if (this.recommendationsSection) {
            this.recommendationsSection.classList.remove('fade-in');
            setTimeout(() => {
                this.recommendationsSection.style.display = 'none';
            }, 300);
        }
    }

    clearCache() {
        this.imageCache.clear();
        this.metadataCache.clear();
        console.log('Caches cleared');
    }

    getCacheStats() {
        return {
            imageCacheSize: this.imageCache.size,
            metadataCacheSize: this.metadataCache.size,
            imageCacheEntries: Array.from(this.imageCache.keys()),
            metadataCacheEntries: Array.from(this.metadataCache.keys())
        };
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing AnimeRecommendationSystem');
    window.animeRecommendationSystem = new AnimeRecommendationSystem();
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimeRecommendationSystem;
}