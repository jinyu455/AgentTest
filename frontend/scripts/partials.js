(function () {
  const PARTIAL_VERSION = "20260614-utf8-restore";

  async function loadPartials() {
    const hosts = [...document.querySelectorAll("[data-partial]")];
    await Promise.all(
      hosts.map(async (host) => {
        const partialPath = host.dataset.partial;
        const separator = partialPath.includes("?") ? "&" : "?";
        const response = await fetch(`${partialPath}${separator}v=${PARTIAL_VERSION}`, {
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`无法加载页面片段：${partialPath}`);
        }
        host.outerHTML = await response.text();
      })
    );
  }

  window.loadPartials = loadPartials;
})();
