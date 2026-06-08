(function () {
  async function loadPartials() {
    const hosts = [...document.querySelectorAll("[data-partial]")];
    await Promise.all(
      hosts.map(async (host) => {
        const partialPath = host.dataset.partial;
        const response = await fetch(partialPath);
        if (!response.ok) {
          throw new Error(`无法加载页面片段：${partialPath}`);
        }
        host.outerHTML = await response.text();
      })
    );
  }

  window.loadPartials = loadPartials;
})();
