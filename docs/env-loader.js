// Environment Variable Loader for Client-Side HTML
// This script loads environment variables from various sources

(function() {
    'use strict';

    // Configuration for different loading methods
    const CONFIG = {
        // Method 1: Load from URL parameters (for development)
        urlParams: true,

        // Method 2: Load from localStorage (user-set values)
        localStorage: true,

        // Method 3: Load from a JSON file (fetch-based)
        jsonFile: 'env.json',

        // Method 4: Load from meta tags (server-injected)
        metaTags: true,

        // Method 5: Prompt user for API keys (fallback)
        userPrompt: false
    };

    // Initialize environment object
    window.ENV = window.ENV || {};

    // Method 1: Load from URL parameters
    function loadFromUrlParams() {
        if (!CONFIG.urlParams) return;

        const urlParams = new URLSearchParams(window.location.search);

        const apiKey = urlParams.get('api_key');
        const clientId = urlParams.get('client_id');

        if (apiKey) window.ENV.GOOGLE_CLOUD_API_KEY = apiKey;
        if (clientId) window.ENV.GOOGLE_CLIENT_ID = clientId;

        // Clean URL after loading (optional)
        if (apiKey || clientId) {
            const cleanUrl = window.location.pathname + window.location.hash;
            window.history.replaceState({}, document.title, cleanUrl);
        }
    }

    // Method 2: Load from localStorage
    function loadFromLocalStorage() {
        if (!CONFIG.localStorage) return;

        try {
            const storedEnv = localStorage.getItem('presentation_env');
            if (storedEnv) {
                const parsed = JSON.parse(storedEnv);
                Object.assign(window.ENV, parsed);
            }
        } catch (e) {
            console.warn('Failed to load environment from localStorage:', e);
        }
    }

    // Method 3: Load from JSON file
    async function loadFromJsonFile() {
        if (!CONFIG.jsonFile) return;

        try {
            const response = await fetch(CONFIG.jsonFile);
            if (response.ok) {
                const data = await response.json();
                Object.assign(window.ENV, data);
                console.log('Loaded environment from JSON file');
            }
        } catch (e) {
            console.warn('Failed to load environment from JSON file:', e);
        }
    }

    // Method 4: Load from meta tags
    function loadFromMetaTags() {
        if (!CONFIG.metaTags) return;

        const apiKeyMeta = document.querySelector('meta[name="google-cloud-api-key"]');
        const clientIdMeta = document.querySelector('meta[name="google-client-id"]');

        if (apiKeyMeta) {
            window.ENV.GOOGLE_CLOUD_API_KEY = apiKeyMeta.getAttribute('content');
        }
        if (clientIdMeta) {
            window.ENV.GOOGLE_CLIENT_ID = clientIdMeta.getAttribute('content');
        }
    }

    // Method 5: Prompt user for API keys (fallback)
    function promptForApiKeys() {
        if (!CONFIG.userPrompt) return;

        // Only prompt if keys are not already loaded
        if (!window.ENV.GOOGLE_CLOUD_API_KEY) {
            const apiKey = prompt('Enter your Google Cloud API Key (optional):');
            if (apiKey) {
                window.ENV.GOOGLE_CLOUD_API_KEY = apiKey;
                // Save to localStorage for future use
                saveToLocalStorage();
            }
        }

        if (!window.ENV.GOOGLE_CLIENT_ID) {
            const clientId = prompt('Enter your Google Client ID (optional):');
            if (clientId) {
                window.ENV.GOOGLE_CLIENT_ID = clientId;
                saveToLocalStorage();
            }
        }
    }

    // Save current ENV to localStorage
    function saveToLocalStorage() {
        try {
            localStorage.setItem('presentation_env', JSON.stringify(window.ENV));
        } catch (e) {
            console.warn('Failed to save environment to localStorage:', e);
        }
    }

    // Utility function to check if environment is loaded
    window.ENV.isLoaded = function() {
        return !!(this.GOOGLE_CLOUD_API_KEY || this.GOOGLE_CLIENT_ID);
    };

    // Utility function to clear stored environment
    window.ENV.clear = function() {
        Object.keys(this).forEach(key => {
            if (key !== 'isLoaded' && key !== 'clear') {
                delete this[key];
            }
        });
        localStorage.removeItem('presentation_env');
    };

    // Load environment variables in order of priority
    async function loadEnvironment() {
        // 1. Load from URL parameters (highest priority)
        loadFromUrlParams();

        // 2. Load from localStorage
        loadFromLocalStorage();

        // 3. Load from JSON file
        await loadFromJsonFile();

        // 4. Load from meta tags
        loadFromMetaTags();

        // 5. Prompt user if nothing loaded and prompting enabled
        if (!window.ENV.isLoaded() && CONFIG.userPrompt) {
            promptForApiKeys();
        }

        // Log loading status
        console.log('Environment loaded:', window.ENV.isLoaded() ? '✅ Keys available' : '⚠️ No keys loaded');
    }

    // Auto-load environment when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadEnvironment);
    } else {
        loadEnvironment();
    }

    // Expose configuration for external modification
    window.EnvLoaderConfig = CONFIG;

})();