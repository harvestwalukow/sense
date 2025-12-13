// SENSE App JavaScript

// Toast initialization
document.addEventListener('DOMContentLoaded', function() {
  const toastElements = document.querySelectorAll('.toast');
  toastElements.forEach(toastEl => {
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
  });

  toastElements.forEach(toastEl => {
    toastEl.addEventListener('hidden.bs.toast', function() {
      toastEl.remove();
    });
  });
});

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});
