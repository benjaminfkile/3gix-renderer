// Optional local settings for the browser renderer. Copy to web/config.js (gitignored)
// and the panel comes up filled in; with autostart the renderer starts on page load.
// The hub's desktop launcher (3GIXHub/scripts/desktop/web.ps1) writes this file for you.
export default {
  hubUrl: "http://localhost:8080",
  apiKey: "local-dev-3gix-key",
  spaceId: "a1000000-0000-0000-0000-000000000001",
  buildId: "",
  timeScale: 3600,
  autostart: true,
};
