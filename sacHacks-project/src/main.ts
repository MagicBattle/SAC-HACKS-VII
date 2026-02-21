import './css/style.css'
import { loadUI } from './css/components/home';

document.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("app");
  if (root) {
    loadUI(root);
  }
});

