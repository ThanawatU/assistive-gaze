// The extension’s brain
// Runs in the background, listens for events, uses Chrome APIs, and connects popup and content scripts.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "HELLO") {
    console.log("Hello from popup 👋");
  }
});
