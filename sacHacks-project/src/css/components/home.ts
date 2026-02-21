export function loadUI(root: HTMLElement) {
    root.innerHTML = `
    <header>
        <div class = "nav-left">
            <img src="/public/icon.svg" alt="SLAP Logo" class="logo" style="width: 40px; height: 40px;">
            <span id = "app-title">SLAP</span>
        </div>

        <nav class="nav-right">
            <button class="nav-button">Home</button>
            <button class="nav-button">About</button>
        </nav>
        
    </header>

    <main class="main-content">
        <div class="welcome-container">
            <h1>SLAP</h1>
            <h3>Sign Language Alphabet Parser</h3>
            <p>SLAP is a tool that helps users learn and recognize sign language alphabets using computer vision.</p>
            <div class="button-container">
                <button class="start-button">Get Started!</button>
                <button class="about-button">Learn More</button>
            </div>
        </div>
    </main>

    <footer class="footer">
        <span class="footer-text">© Copyright</span>
    </footer>
    `
}