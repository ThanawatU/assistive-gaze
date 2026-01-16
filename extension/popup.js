document.getElementById("alert_test").onclick = () => {
     chrome.runtime.sendMessage({ action: "HELLO" });
    alert("test");
};
