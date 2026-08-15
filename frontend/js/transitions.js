// Shared page-load and navigation transitions using GSAP

function pageEnter() {
    gsap.set("body", { opacity: 1 });

    const navbar = document.querySelector(".navbar");
    if (navbar) {
        gsap.from(navbar, { y: -24, opacity: 0, duration: 0.5, ease: "power2.out" });
    }

    const cards = document.querySelectorAll(".card");
    gsap.from(cards, {
        y: 20,
        opacity: 0,
        duration: 0.55,
        ease: "power2.out",
        stagger: 0.08,
        delay: navbar ? 0.15 : 0,
    });
}

function navigateWithTransition(url) {
    gsap.to("body", {
        opacity: 0,
        duration: 0.28,
        ease: "power1.in",
        onComplete: () => { window.location.href = url; },
    });
}

function bounceClick(el) {
    gsap.fromTo(el, { scale: 0.96 }, { scale: 1, duration: 0.25, ease: "back.out(3)" });
}

function revealResult(el) {
    gsap.fromTo(
        el,
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, ease: "power2.out", clearProps: "transform" }
    );
}

document.addEventListener("DOMContentLoaded", () => {
    gsap.set("body", { opacity: 0 });
    pageEnter();
});