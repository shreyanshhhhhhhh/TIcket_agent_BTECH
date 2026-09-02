// Shared page-load and navigation transitions + Toast & Micro-interactions

function pageEnter() {
    gsap.set("body", { opacity: 1 });

    const navbar = document.querySelector(".navbar");
    if (navbar) {
        gsap.from(navbar, { y: -20, opacity: 0, duration: 0.45, ease: "power3.out" });
    }

    const cards = document.querySelectorAll(".card, .sidebar-card, .login-card");
    if (cards.length) {
        gsap.from(cards, {
            y: 22,
            opacity: 0,
            duration: 0.5,
            ease: "power3.out",
            stagger: 0.06,
            delay: navbar ? 0.1 : 0,
        });
    }
}

function navigateWithTransition(url) {
    gsap.to("body", {
        opacity: 0,
        y: -10,
        duration: 0.24,
        ease: "power2.in",
        onComplete: () => { window.location.href = url; },
    });
}

function bounceClick(el) {
    if (!el) return;
    gsap.fromTo(el, { scale: 0.94 }, { scale: 1, duration: 0.25, ease: "back.out(2.5)" });
}

function revealResult(el) {
    if (!el) return;
    gsap.fromTo(
        el,
        { y: 20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.45, ease: "power3.out", clearProps: "transform" }
    );
}

// ─── Toast System ──────────────────────────────────────────
function showToast(message, type = 'info') {
    let container = document.getElementById('toastContainer');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toastContainer';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const iconMap = {
        success: '✅',
        warning: '⚠️',
        error: '❌',
        info: '✨'
    };

    toast.innerHTML = `
        <span style="font-size:17px;line-height:1;">${iconMap[type] || 'ℹ️'}</span>
        <div style="flex:1;">${message}</div>
    `;

    container.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    // Auto remove after 3.5s
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 320);
    }, 3500);
}

// ─── Confetti Celebration ─────────────────────────────────
function triggerConfetti() {
    const colors = ['#00f2fe', '#4facfe', '#a855f7', '#10b981', '#fbbf24'];
    const count = 35;

    for (let i = 0; i < count; i++) {
        const conf = document.createElement('div');
        conf.style.position = 'fixed';
        conf.style.zIndex = '99999';
        conf.style.width = Math.random() * 8 + 6 + 'px';
        conf.style.height = Math.random() * 5 + 4 + 'px';
        conf.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        conf.style.left = '50vw';
        conf.style.top = '40vh';
        conf.style.borderRadius = '2px';
        conf.style.pointerEvents = 'none';
        document.body.appendChild(conf);

        const angle = Math.random() * Math.PI * 2;
        const velocity = Math.random() * 220 + 80;
        const destX = Math.cos(angle) * velocity;
        const destY = Math.sin(angle) * velocity - 60;

        gsap.to(conf, {
            x: destX,
            y: destY,
            rotation: Math.random() * 720 - 360,
            opacity: 0,
            duration: Math.random() * 0.8 + 0.7,
            ease: "power2.out",
            onComplete: () => conf.remove()
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    gsap.set("body", { opacity: 0 });
    pageEnter();
});